# geracao.py – Blueprint de geração de imagens (Studio Elle)
from __future__ import annotations
import json, logging, os, random, re, threading, unicodedata, time, shutil
from typing import Any, Optional
from functools import lru_cache

import torch, torchvision.transforms as T
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
)
from flask import (
    Blueprint, flash, jsonify,
    redirect, render_template,
    request, url_for,
)

from auth_utils import login_required
from config_utils import (
    get_directories_indicizzate,
    get_generative_dirs,
    set_generative_dirs,
)
from modello import cerca_immagini

# ───────── Config índice FAISS ─────────
FILE_INDICE_FAISS = "indice_faiss.index"
FILE_PERCORSI     = "percorsi_immagini.pkl"

# ───────────── LOG / BP ─────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s in %(module)s: %(message)s"))
logger.addHandler(handler)

geracao_bp = Blueprint("geracao_bp", __name__)

# ─────────── HARDWARE ───────────
device = "cuda" if torch.cuda.is_available() else "cpu"
VRAM_GB = (
    torch.cuda.get_device_properties(0).total_memory / 1024**3
    if device == "cuda"
    else 0
)
LOW_VRAM = VRAM_GB < 6

# ───────── Safety checker NOP ─────────
def _noop_safety(*args: Any, **kw: Any):
    if args and args[0] is not None:
        imgs = args[0]
    elif kw.get("images") is not None:
        imgs = kw["images"]
    elif kw.get("imgs") is not None:
        imgs = kw["imgs"]
    else:
        raise ValueError("Imagens não encontradas em _noop_safety")
    n = imgs.shape[0] if hasattr(imgs, "shape") else len(imgs)
    return imgs, [False] * n

# ───────── PROMPTS / CONSTS ─────────
NEG_PROMPT_BASE = (
    "blurry, low quality, lowres, jpeg artifacts, noisy, boring, low contrast, "
    "bad anatomy, watermark, grainy, text"
)
PROMPT_SNIPPETS = [
    "paisley elegante, colori vivaci",
    "floreale acquerello soft, primavera",
    "geometrico minimalista, due toni",
    "texture denim indigo, realistica",
]

# ───────── CATEGORIE GEN ─────────
CATEGORIE_FILE_GEN = "static/categorie_generativa.json"
def _carica_categorie_gen() -> dict:
    try:
        with open(CATEGORIE_FILE_GEN, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
CATEGORIE_GEN = _carica_categorie_gen()

# ───────── SLUGIFY ─────────
def slugify(txt: str, maxlen: int = 40) -> str:
    txt = unicodedata.normalize("NFKD", txt).encode("ascii","ignore").decode()
    txt = re.sub(r"[^a-zA-Z0-9_-]+","_", txt.lower()).strip("_")
    return (txt or "image")[:maxlen]

# ─── Busca imagem indexada ───
def _busca_imagem_indexada(query: str) -> Optional[str]:
    try:
        dirs = get_directories_indicizzate()
        dirs_sel = [d["path"] if isinstance(d,dict) else d for d in dirs]
        if not (dirs_sel and query):
            return None
        imgs, _ = cerca_immagini(
            descrizione=query,
            categoria="",
            file_indice_faiss=FILE_INDICE_FAISS,
            file_percorsi=FILE_PERCORSI,
            directories_selezionate=dirs_sel,
            offset=0,
            limit=1,
        )
        if imgs:
            return imgs[0]["percorso"]
    except Exception as exc:
        logger.warning("Falha na busca indexada: %s", exc)
    return None

# ───────── PIPELINES ─────────
def check_gpu_available() -> bool:
    return torch.cuda.is_available()

# ---------------- PIPELINE CACHE & LOADER --------------------------
@lru_cache(maxsize=1)
def _get_pipe():
    """
    Devolve (pipe, None) sempre.  *Upscaler removido*.
    • carrega localmente se já houver pasta cacheada em ./models/sd15
    • senão tenta baixar online usando variant="fp16"
    • se falhar, tenta novamente em fp32
    """
    if not check_gpu_available():
        raise RuntimeError("GPU non disponibile")

    model_id = "runwayml/stable-diffusion-v1-5"
    local_dir = "./models/sd15"                    # opcional: pasta local

    def _try_load(**kw):
        return StableDiffusionPipeline.from_pretrained(
            local_dir if os.path.isdir(local_dir) else model_id,
            **kw,
            safety_checker=None
        )

    try:
        logger.info("[GEN] Carregando SD-1.5 em GPU (fp16, low_cpu_mem_usage=True)")
        pipe = _try_load(
            torch_dtype=torch.float16,
            variant="fp16",
            low_cpu_mem_usage=True
        )
    except Exception as e_fp16:
        logger.warning("[GEN] fp16 indisponível – %s", e_fp16)
        logger.info("[GEN] Tentando fp32…")
        pipe = _try_load(
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

    pipe.to("cuda")
    pipe.enable_attention_slicing()
    pipe.safety_checker = None        # NOP

    # upscaler REMOVIDO (para economizar VRAM)
    return pipe, None
# ─── ControlNet opcional (inalterado) ───
_pipe_cn: Optional[StableDiffusionControlNetPipeline] = None
def _load_controlnet():
    global _pipe_cn
    if _pipe_cn or not check_gpu_available():
        return _pipe_cn
    logger.info("[GEN] Carregando ControlNet em GPU fp16")
    try:
        cn = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11f1e_sd15_tile",
            torch_dtype=torch.float16
        )
        pipe_cn = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=cn,
            safety_checker=None,
            torch_dtype=torch.float16
        )
        pipe_cn.to("cuda")
        pipe_cn.enable_attention_slicing()
        _pipe_cn = pipe_cn
    except Exception as e:
        logger.warning("[GEN] Falha carregando ControlNet: %s", e)
    return _pipe_cn

# ───────── GALERIA RECENTE ─────────
def _ultime(n=20):
    g = "static/generated"
    if not os.path.isdir(g):
        return []
    fs = [f for f in os.listdir(g) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    fs.sort(key=lambda x: os.path.getmtime(os.path.join(g, x)), reverse=True)
    return fs[:n]

# ───────── PROGRESSO ─────────
gen_lock = threading.Lock()
gen_status = {"running": False, "percent": 0, "done": False}

# ───────── ROTA PRINCIPAL ─────────
@geracao_bp.route("/geracao", methods=["GET","POST"], endpoint="geracao")
@login_required
def geracao():
    if request.method == "POST":
        if not check_gpu_available():
            flash("GPU non disponibile – impossibile generare.", "error")
            return redirect(url_for("geracao"))

        f = request.form
        prompt_raw = f.get("prompt", "").strip()
        cat_sel    = f.get("categoria", "").strip()
        tags_raw   = f.get("tags_selezionate", "").strip()

        logger.info("Iniciando geração… prompt=%r cat=%r tags=%r",
                    prompt_raw, cat_sel, tags_raw)

        # -------- imagem base (upload ou indexada) --------
        base_file = request.files.get("base_image")
        init_image = None
        if base_file and base_file.filename:
            try:
                init_image = Image.open(base_file.stream).convert("RGB")
            except Exception as e:
                logger.warning("Erro lendo imagem-base: %s", e)

        if init_image is None:
            cat_desc = CATEGORIE_GEN.get(cat_sel, {}).get("descrizione", "")
            query = " ".join([prompt_raw, cat_desc, tags_raw.replace(",", " ")]).strip()
            match_path = _busca_imagem_indexada(query)
            if match_path:
                try:
                    init_image = Image.open(match_path).convert("RGB")
                    logger.info("Usando imagem indexada: %s", match_path)
                except Exception as e:
                    logger.warning("Erro abrindo indexada: %s", e)

        if init_image is None:
            flash("Nessuna immagine fornita o trovata.", "error")
            return redirect(url_for("geracao"))

        # -------- prompt completo --------
        prompt_full = " ".join(filter(None, [
            CATEGORIE_GEN.get(cat_sel, {}).get("descrizione", ""),
            tags_raw,
            prompt_raw,
        ])) or "fabric pattern"
        negative_full = NEG_PROMPT_BASE

        # -------- resolução seguro-1050Ti --------
        ris = f.get("risoluzione", "512x512")
        if ris == "custom":
            w, h = int(f.get("custom_w",512)), int(f.get("custom_h",512))
        else:
            w, h = map(int, ris.split("x"))
        w, h = (max(256,min(w,4096))//8*8, max(256,min(h,4096))//8*8)
        if w > 512 or h > 512:
            logger.info("Res >512×512 forçada para 512×512 (GPU modesta)")
            w = h = 512

        steps = int(f.get("steps",30))
        cfg   = float(f.get("cfg",8))
        seed  = random.randint(0,2**32-1) if int(f.get("seed",-1))<0 else int(f.get("seed"))
        gen_t = torch.Generator("cuda").manual_seed(seed)

        # -------- pipeline --------
        try:
            pipe = _get_pipe()                     # ← só ele
        except RuntimeError as e:
            flash(str(e), "error")
            return redirect(url_for("geracao"))

        # -------- scheduler --------
        pipe.scheduler = {
            "euler_a": EulerAncestralDiscreteScheduler,
            "dpm++_2m": DPMSolverMultistepScheduler,
            "ddim": DDIMScheduler,
        }.get(f.get("sampler","dpm++_2m"), DPMSolverMultistepScheduler).from_config(
            pipe.scheduler.config, use_karras_sigmas=True
        )
        pipe.enable_vae_tiling()

        # -------- progresso --------
        with gen_lock:
            gen_status.update({"running":True,"percent":0,"done":False})
        def _cb(i, *_):
            with gen_lock:
                gen_status["percent"] = int(i/steps*90)

        # -------- gera --------
        try:
            tensor_img = T.ToTensor()(init_image).unsqueeze(0)*2 - 1
            vae_param  = next(pipe.vae.parameters())
            tensor_img = tensor_img.to(vae_param.device, dtype=vae_param.dtype)
            latents    = pipe.vae.encode(tensor_img).latent_dist.sample() * 0.18215
            latents    = latents.to(next(pipe.unet.parameters()).dtype)

            out = pipe(
                prompt=prompt_full,
                negative_prompt=negative_full,
                latents=latents,
                strength=0.6,
                width=w, height=h,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=gen_t,
                callback=_cb, callback_steps=1
            )
            img = out.images[0]
            with gen_lock:
                gen_status.update({"percent":100,"done":True})
        except Exception as e:
            logger.error("Erro na geração: %s", e, exc_info=True)
            flash(f"Erro durante la generazione: {e}", "error")
            with gen_lock:
                gen_status.update({"running":False,"done":True})
            return redirect(url_for("geracao"))

        # -------- salva --------
        outdir = "static/generated"
        os.makedirs(outdir, exist_ok=True)
        fname = slugify(prompt_raw or cat_sel or "image")
        if tags_raw:
            fname += f"__{slugify(tags_raw)}"
        fname += ".png"
        try:
            img.save(os.path.join(outdir, fname))
        except Exception as e:
            logger.error("Falha ao salvar: %s", e)
            flash(f"Erro salvando immagine: {e}", "error")
            return redirect(url_for("geracao"))

        flash("Immagine generata!", "success")
        return render_template(
            "geracao.html",
            generated_image=url_for("static", filename=f"generated/{fname}"),
            ultimas_imagens=_ultime(),
            snippets=PROMPT_SNIPPETS,
            categorie=CATEGORIE_GEN,
            low_vram=LOW_VRAM,
            tile_preview_selected=False,
        )

    # GET
    return render_template(
        "geracao.html",
        generated_image=None,
        ultimas_imagens=_ultime(),
        snippets=PROMPT_SNIPPETS,
        categorie=CATEGORIE_GEN,
        low_vram=LOW_VRAM,
        tile_preview_selected=False,
    )

# ───────── STATUS JSON ─────────
@geracao_bp.route("/stato_generazione")
@login_required
def stato_generazione():
    with gen_lock:
        return jsonify(gen_status.copy())

# ───────── (config dirs, LoRA, …) ─────────
# TODO: resto do arquivo permanece **idêntico** ao seu,
# pois o upscaler só era usado nestas seções acima.


# ───────── CONFIG DIR. GENERATIVI ─────────
# (permane igual ao original)
@geracao_bp.route("/configurazione_generativa", methods=["GET","POST"], endpoint="configurazione_generativa")
@login_required
def configurazione_generativa():
    dirs_ind = get_directories_indicizzate()
    cfg = get_generative_dirs()
    if request.method == "POST":
        new_cfg = {
            (d["path"] if isinstance(d,dict) else d): f"enable_{d['path'] if isinstance(d,dict) else d}" in request.form
            for d in dirs_ind
        }
        set_generative_dirs(new_cfg)
        flash("Configurazione salvata!", "success")
        return redirect(url_for("configurazione_generativa"))
    return render_template(
        "configurazione_generativa.html",
        directories_indicizzate=dirs_ind,
        generative_data=cfg,
    )

# ───────── Treino LoRA ───────────────────
# (permane igual ao original)
lora_lock = threading.Lock()
lora_progress = {"running": False, "percent": 0, "log": [], "completed": False}
lora_thread: Optional[threading.Thread] = None

@geracao_bp.route("/treino_lora", methods=["POST"], endpoint="treino_lora")
@login_required
def treino_lora():
    global lora_thread
    cfg = get_generative_dirs()
    sel = [d for d, enabled in cfg.items() if enabled]
    if not sel:
        flash("Nessun direttorio selezionato!", "error")
        return redirect(url_for("configurazione_generativa"))

    tmp_data = f"/tmp/train_lora_{int(time.time())}"
    os.makedirs(tmp_data, exist_ok=True)
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    for d in sel:
        for root, _, files in os.walk(d):
            for f_ in files:
                if f_.lower().endswith(exts):
                    shutil.copy2(os.path.join(root, f_), os.path.join(tmp_data, f_))

    outdir = f"./lora_output_{random.randint(1000,9999)}"
    cmd = [
        "accelerate", "launch", "train_lora.py",
        "--pretrained_model_name=runwayml/stable-diffusion-v1-5",
        f"--train_data_dir={tmp_data}",
        f"--output_dir={outdir}",
        "--instance_prompt=fabricStyle",
        "--resolution=512",
        "--train_batch_size=1",
        "--gradient_accumulation_steps=1",
        "--learning_rate=1e-4",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        "--max_train_steps=1000",
    ]

    with lora_lock:
        lora_progress.update({"running": True, "percent": 0, "log": [], "completed": False})

    def _run_lora():
        import subprocess, re
        step_re = re.compile(r"Step (\d+)/(\d+)")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                with lora_lock:
                    lora_progress["log"].append(line.strip())
                    m = step_re.search(line)
                    if m:
                        cur, tot = map(int, m.groups()); lora_progress["percent"] = int(cur/tot*100)
        code = proc.wait()
        with lora_lock:
            lora_progress.update({"completed": True, "running": False})
            lora_progress["log"].append("Concluído" if code==0 else f"Erro code={code}")

    lora_thread = threading.Thread(target=_run_lora, daemon=True)
    lora_thread.start()

    flash("Treinamento LoRA avviato!", "info")
    return redirect(url_for("configurazione_generativa"))

@geracao_bp.route("/stato_lora", methods=["GET"], endpoint="stato_lora")
@login_required
def stato_lora():
    with lora_lock:
        return jsonify(lora_progress.copy())    #!/usr/bin/env python3
# ... O restante do arquivo permanece inalterado, mantendo toda a lógica original de indexação, conversor, etc.
# Apenas as primeiras partes (carregamento pipeline e view geracao) foram estendidas com GPU checks.

# Note: garanta que você reinicie o servidor após salvar estas mudanças, para que _get_pipe.cache_clear() e reload funcionem corretamente.
