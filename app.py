import os, threading, json, subprocess, pickle, random, tempfile
import unicodedata, re, logging, warnings
from collections import Counter
from io import BytesIO
from urllib.parse import unquote
from logging.handlers import TimedRotatingFileHandler
from threading import Lock
from threading import Lock
import json, os, time
import zipfile
import numpy as np
import faiss

from PIL import Image
from flask import (
    Flask, flash, jsonify, redirect, render_template, request,
    send_file, session, url_for
)
CONFIG_FILE = "config.json"

# ==================== HELPERS DE DIRETÓRIO (sempre aplicar) ====================
def _path_in_dirs(path: str, dirs: list[str]) -> bool:
    """Retorna True se 'path' pertence a qualquer diretório em 'dirs'."""
    if not dirs:
        return True
    try:
        pn = os.path.normpath(path)
        for d in dirs:
            dn = os.path.normpath(d)
            # começa pelo dir (aceita igual ou dentro)
            if pn == dn or pn.startswith(dn + os.sep) or pn.startswith(dn):
                return True
    except Exception:
        pass
    return False

def _filter_paths_by_dirs(paths: list[str], dirs: list[str]) -> list[str]:
    """Filtra a lista completa de caminhos para conter só o que está dentro de 'dirs'."""
    if not dirs:
        return list(paths)
    out = []
    for p in paths:
        if _path_in_dirs(p, dirs):
            out.append(p)
    return out

# ----------------- Conversion Options Persistenti -----------------
def load_conversion_options():
    cfg = load_config()
    return cfg.get("conversion_options", {
        "width": None,
        "height": None,
        "square": True,
        "keep_aspect": True,
        "output_format": "jpg",
        "skip_duplicates": True,
        "workers": os.cpu_count(),
        "quality": 85,
        "colorize_ia": False
    })

def save_conversion_options(opts):
    cfg = load_config()
    cfg["conversion_options"] = opts
    save_config(cfg)

def load_config():
    if not os.path.exists(CONFIG_FILE):
        return {}
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_config(cfg):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[ERRO] salvando config: {e}")

# Blueprints / geração
from geracao import geracao_bp, _get_pipe
# Subir pipeline de geração **só se** explicitamente habilitado
if os.getenv("BYTEAI_GENERATION", "off").lower() in ("1", "true", "on", "yes"):
    threading.Thread(target=_get_pipe, daemon=True).start()


# Projeto
from auth_utils import login_required
from progress import indicizzazione_progress, progress_lock
from indicizza import indicizza_immagini
from modello import cerca_immagini, encontrar_duplicatas, extrai_features_imagem, cerca_per_embedding
from werkzeug.utils import secure_filename
from apscheduler.schedulers.background import BackgroundScheduler

# -----------------------------------------------------------------------
#  VARI AMBIENTE
# -----------------------------------------------------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"

# -----------------------------------------------------------------------
#  LOG
# -----------------------------------------------------------------------
for noisy in (
    "PIL", "urllib3", "pyvips", "diffusers", "transformers",
    "huggingface_hub", "accelerate",
):
    logging.getLogger(noisy).setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = TimedRotatingFileHandler(
    os.path.join(log_dir, "app.log"),
    when="midnight", backupCount=30, encoding="utf-8"
)
handler.setFormatter(
    logging.Formatter("[%(asctime)s] %(levelname)s in %(module)s: %(message)s")
)
logger.addHandler(handler)



# -----------------------------------------------------------------------
# FLASK APP
# -----------------------------------------------------------------------

app = Flask(__name__)
app.secret_key = "chiave_segreta_per_flash_message"

# Registriamo il blueprint
app.register_blueprint(geracao_bp)
FILTERS_KEY = "filtros_galeria"          # nome na sessão
STATS_FILE = "search_stats.json"
stats_lock = threading.Lock()
### HELPRS
# ====== Sentinela di conversione: helpers globali ======
WATCH_EXTS = (".psd", ".psb", ".ifd", ".tif", ".tiff",
              ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".jp2")

def _load_indexed_paths():
    """Carica i percorsi già indicizzati (percorsi_immagini.pkl) per saltare output già nel CLIP."""
    try:
        if os.path.exists(file_percorsi):
            with open(file_percorsi, "rb") as f:
                return set(pickle.load(f))
    except Exception:
        pass
    return set()

def _run_one_scan():
    from datetime import datetime
    import pathlib

    cfg = load_config()
    regole = cfg.get("conversion_watch", [])
    if not regole:
        _auto_log("[SCAN] Nessuna regola configurata.")
        with auto_conv_lock:
            s = auto_conversion_status.setdefault("stats", {})
            s.setdefault("last_scan", "-")
        return

    # extensões suportadas
    WATCH_EXTS = (".psd", ".psb", ".ifd", ".tif", ".tiff",
                  ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".jp2")

    # --- carrega todos os caminhos indexados (imagens finais) uma volta ---
    indexed_paths: set[str] = set()
    try:
        if os.path.exists(file_percorsi):
            with open(file_percorsi, "rb") as f:
                indexed_paths = set(pickle.load(f))
    except Exception:
        indexed_paths = set()

    # mapa por diretório de saída -> nomes de arquivo já indexados lá
    indexed_by_outdir: dict[str, set[str]] = {}
    for p in indexed_paths:
        p_norm = os.path.normpath(p)
        outdir = os.path.dirname(p_norm)
        fname = os.path.basename(p_norm)
        indexed_by_outdir.setdefault(outdir, set()).add(fname)

    # helpers de coerção segura
    def _to_int_or_none(v):
        if v in (None, "", "None"):
            return None
        try:
            return int(v)
        except Exception:
            return None

    def _to_bool(v):
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in ("1", "true", "on", "yes", "y", "t")

    # existência robusta (ignora arquivos 0 bytes/corrompidos)
    def _file_exists_ok(path: str) -> bool:
        try:
            if not os.path.isfile(path):
                return False
            return os.path.getsize(path) > 1024  # >1KB
        except Exception:
            return False

    local_scanned = local_converted = local_skip_ex = local_skip_idx = local_err = 0

    for rule in regole:
        in_dir = rule.get("input_dir")
        out_dir = rule.get("output_dir")
        opts = (rule.get("options") or {})

        if not in_dir or not out_dir:
            continue
        if not os.path.exists(in_dir):
            _auto_log(f"[WARN] Input non esiste: {in_dir}")
            continue

        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            _auto_log(f"[ERRORE] Output non accessibile: {out_dir} ({e})")
            local_err += 1
            continue

        # --- prepara opções do conversor ---
        try:
            from conversor import ConversionOptions, _process

            ctor_kwargs = {}
            if "width" in opts:   ctor_kwargs["width"] = _to_int_or_none(opts.get("width"))
            if "height" in opts:  ctor_kwargs["height"] = _to_int_or_none(opts.get("height"))
            if "workers" in opts: ctor_kwargs["workers"] = int(opts.get("workers") or 1)

            if "square" in opts:      ctor_kwargs["square"] = _to_bool(opts.get("square"))
            if "keep_aspect" in opts: ctor_kwargs["keep_aspect"] = _to_bool(opts.get("keep_aspect"))

            ctor_kwargs["skip_duplicates"] = True
            if "icc_profile" in opts:   ctor_kwargs["icc_profile"] = opts.get("icc_profile") or None
            if "output_format" in opts: ctor_kwargs["output_format"] = str(opts.get("output_format") or "jpg").lower()
            else:                       ctor_kwargs["output_format"] = "jpg"

            conv_opts = ConversionOptions(**ctor_kwargs)

            if "colorize_ia" in opts and hasattr(conv_opts, "colorize_ia"):
                setattr(conv_opts, "colorize_ia", _to_bool(opts.get("colorize_ia")))
            if "quality" in opts and hasattr(conv_opts, "quality"):
                q = _to_int_or_none(opts.get("quality"))
                if q is not None:
                    try:
                        setattr(conv_opts, "quality", int(q))
                    except Exception:
                        pass

        except Exception as e:
            _auto_log(f"[ERRORE] Opzioni non valide per la regola {in_dir} → {out_dir}: {e}")
            local_err += 1
            continue

        indexed_names_here = indexed_by_outdir.get(os.path.normpath(out_dir), set())

        for root, _, files in os.walk(in_dir):
            for fname in files:
                if not fname.lower().endswith(WATCH_EXTS):
                    continue

                src = os.path.join(root, fname)
                base = os.path.splitext(fname)[0]
                out = os.path.join(out_dir, base + f".{conv_opts.output_format}")

                local_scanned += 1

                # 1) já existe na pasta de saída
                if _file_exists_ok(out):
                    local_skip_ex += 1
                    _auto_log(f"[SKIP esiste] {src} → {out}")
                    continue

                # 2) já indexado nesse out_dir
                if os.path.basename(out) in indexed_names_here:
                    local_skip_idx += 1
                    _auto_log(f"[SKIP indicizzato] {src} → {out}")
                    continue

                # 3) tenta converter
                try:
                    status, path_ret, detail = _process(src, conv_opts, out_dir)
                    if status == "converted":
                        local_converted += 1
                        _auto_log(f"[OK] {src} → {out}")
                    elif status == "skipped":
                        local_skip_ex += 1
                        _auto_log(f"[SKIP esiste] {src} → {out}")
                    else:
                        local_err += 1
                        _auto_log(f"[ERRORE] {src}: {detail or 'errore sconosciuto'}")
                except Exception as e:
                    local_err += 1
                    _auto_log(f"[ERRORE] {src}: {e}")

    with auto_conv_lock:
        s = auto_conversion_status.setdefault("stats", {})
        s["scanned"] = s.get("scanned", 0) + local_scanned
        s["converted"] = s.get("converted", 0) + local_converted
        s["skipped_existing"] = s.get("skipped_existing", 0) + local_skip_ex
        s["skipped_indexed"] = s.get("skipped_indexed", 0) + local_skip_idx
        s["errors"] = s.get("errors", 0) + local_err
        s["last_scan"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# -------------------------------------------------------------------
# -------------------------------------------------------------------
# OCR (na RAIZ do projeto)
# -------------------------------------------------------------------
OCR_META_FILE = "ocr_metadata.json"

def carrega_ocr_metadata() -> dict[str, str]:
    """Carrega (ou cria vazio) o arquivo JSON com texto/tag extraído via OCR na RAIZ."""
    if not os.path.exists(OCR_META_FILE):
        with open(OCR_META_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)
        return {}
    try:
        with open(OCR_META_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Errore lendo %s: %s", OCR_META_FILE, e)
        return {}

def _load_stats():
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}
import unicodedata, re
def _norm(s: str) -> str:
    s = unicodedata.normalize('NFKC', s)
    s = s.casefold()
    s = re.sub(r'[^\w\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()



def _save_stats(d):                         # d = Counter  (imagem->hits)
    with stats_lock:
        with open(STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)

INFO_IMMAGINI_FILE = "info_immagini.json"

def carica_info_immagini() -> dict:
    if not os.path.exists(INFO_IMMAGINI_FILE):
        with open(INFO_IMMAGINI_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)
        return {}
    try:
        with open(INFO_IMMAGINI_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Erro lendo %s: %s", INFO_IMMAGINI_FILE, e)
        return {}

def salva_info_immagini(data: dict):
    try:
        with open(INFO_IMMAGINI_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error("Erro salvando %s: %s", INFO_IMMAGINI_FILE, e)
# -----------------------------------------------------------------------
#  *** ALIAS AUTOMATICI ***  per mantenere i vecchi url_for()
# -----------------------------------------------------------------------

def _crea_alias(bp):
    for rule in list(app.url_map.iter_rules()):
        if rule.endpoint.startswith(bp.name + "."):
            alias = rule.endpoint.split(".", 1)[1]
            if alias not in app.view_functions:
                app.add_url_rule(
                    rule.rule,
                    endpoint=alias,
                    view_func=app.view_functions[rule.endpoint],
                    methods=list(rule.methods),
                )
                logger.debug("Alias %s → %s", alias, rule.rule)

_crea_alias(geracao_bp)


# ========================= CONFIGURAÇÃO DO LOGGER =========================
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from logging.handlers import TimedRotatingFileHandler
handler = TimedRotatingFileHandler(
    os.path.join(log_dir, "app.log"),
    when="midnight",
    interval=1,
    backupCount=30,
    encoding="utf-8"
)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s in %(module)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

logging.basicConfig(level=logging.DEBUG)

# ========================= ARQUIVOS E CONFIGS =========================
file_indice_faiss = 'indice_faiss.index'
file_percorsi = 'percorsi_immagini.pkl'
embeddings_file = 'embeddings_immagini.npy'
config_file = 'config.json'
categorie_file = 'static/categorie.json'
# === Conversão Automática: estado/log/stats ===
auto_conv_lock = Lock()
auto_conversion_status = {
    "log": [],           # últimas mensagens
    "running": False,    # se o sentinela está no ciclo
    "stats": {           # counters acumulados
        "scanned": 0,
        "converted": 0,
        "skipped_existing": 0,
        "skipped_indexed": 0,
        "errors": 0,
        "last_scan": "-"
    }
}

def _auto_log(msg):
    from datetime import datetime
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    with auto_conv_lock:
        auto_conversion_status["log"].append(line)
        # mantém curto
        if len(auto_conversion_status["log"]) > 300:
            auto_conversion_status["log"] = auto_conversion_status["log"][-300:]

# ==================== LOCKS / GLOBAIS ====================
conv_lock = Lock()
stats_lock = Lock()
info_lock = Lock()
schedule_lock = Lock()
reindicizzazione_lock = Lock()
eliminazione_lock = Lock()
encontrar_lock = Lock()


indicizzazione_thread = None
indicizzazione_interrompida = threading.Event()
directories_indicizzate = []
# Estado global do conversor
conv_lock = Lock()
conversion_status = {
    "log": [],
    "progress": 0,
    "running": False,
    "sent": 0
}


# ==================== HELPERS: JSON ROBUSTO ====================
def _load_json_robusto(path, default):
    """
    Tenta ler JSON em utf-8; se falhar, tenta cp1252.
    Se continuar falhando, faz backup .old e recria com 'default'.
    """
    if not os.path.exists(path):
        return default
    # 1) UTF-8
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    # 2) CP1252 (Windows)
    try:
        with open(path, 'r', encoding='cp1252') as f:
            return json.load(f)
    except Exception:
        # 3) backup e recria limpo
        try:
            os.replace(path, path + '.old')
        except Exception as e:
            logger.error("Falha ao fazer backup de %s: %s", path, e)
        with open(path, 'w', encoding='utf-8') as fw:
            json.dump(default, fw, ensure_ascii=False, indent=4)
        return default

def _atomic_write_json(path, data):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    os.replace(tmp, path)

# ==================== STATS ====================
FILTERS_KEY = "filtros_galeria"
STATS_FILE = "search_stats.json"

def _load_stats():
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _save_stats(d):
    with stats_lock:
        with open(STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)

def _norm(s: str) -> str:
    s = unicodedata.normalize('NFKC', s)
    s = s.casefold()
    s = re.sub(r'[^\w\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

# ==================== CONFIG (dirs/utenti/schedule) ====================
def carica_configurazione():
    if not os.path.exists(config_file):
        return []
    with open(config_file, 'r') as f:
        try:
            config_data = json.load(f)
        except json.JSONDecodeError:
            return []
    dirs = config_data.get('directories_indicizzate', [])
    new_list = []
    for d in dirs:
        if isinstance(d, str):
            new_list.append({"path": d, "nome": ""})
        elif isinstance(d, dict):
            new_list.append({"path": d.get("path",""), "nome": d.get("nome","")})
    return new_list

def salva_configurazione(directories_indicizzate_local):
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config_completo = json.load(f)
        except json.JSONDecodeError:
            config_completo = {}
    else:
        config_completo = {}
    config_completo['directories_indicizzate'] = directories_indicizzate_local
    with open(config_file, 'w') as f:
        json.dump(config_completo, f, ensure_ascii=False, indent=4)

try:
    directories_indicizzate = carica_configurazione()
except Exception as e:
    logger.error("[ERROR] Carregando configurações: %s", e)
    directories_indicizzate = []

# ========== Funções de config generativa (usadas por “geracao.py”, mas deixamos aqui) ==========

def carica_configurazione_generativa():
    if not os.path.exists(config_file):
        return {}
    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        return config_data.get("generative_dirs", {})
    except:
        return {}

def salva_configurazione_generativa(generative_data):
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config_completo = json.load(f)
        except json.JSONDecodeError:
            config_completo = {}
    else:
        config_completo = {}
    config_completo["generative_dirs"] = generative_data
    with open(config_file, 'w') as f:
        json.dump(config_completo, f, ensure_ascii=False, indent=4)

# ==================== Carregar e Salvar Usuários ====================
def carica_utenti():
    if not os.path.exists(config_file):
        return []
    try:
        with open(config_file, 'r') as f:
            data = json.load(f)
        return data.get('users', [])
    except:
        return []

def salva_utenti(users_list):
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config_completo = json.load(f)
        except json.JSONDecodeError:
            config_completo = {}
    else:
        config_completo = {}
    config_completo['users'] = users_list
    with open(config_file, 'w') as f:
        json.dump(config_completo, f, ensure_ascii=False, indent=4)

def ensure_default_user():
    users = carica_utenti()
    found = False
    for u in users:
        if u.get('username') == 'chickellero':
            found = True
            break
    if not found:
        users.append({
            'username': 'chickellero',
            'password': 'chickellero',
            'nome': 'Marco'
        })
        salva_utenti(users)
        logger.info("Usuário padrão 'chickellero' adicionado ao config.json.")

# Carrega categorias
try:
    def carica_categorie():
        if not os.path.exists(categorie_file):
            with open(categorie_file, 'w') as f:
                json.dump({}, f, ensure_ascii=False, indent=4)
            return {}
        with open(categorie_file, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    categorie = carica_categorie()
except Exception as e:
    logger.error("[ERROR] Carregando categorias: %s", e)
    categorie = {}

# Progressos para duplicatas, reindex e etc.
eliminazione_progress = {'percentuale': 0, 'log': [], 'completed': False}
eliminazione_lock = threading.Lock()

encontrar_progress = {
    'percentuale': 0,
    'log': [],
    'completed': False,
    'grupos_duplicatas': None,
    'similaridade_valor': 0,
    'total_duplicatas': 0
}
encontrar_lock = threading.Lock()

reindicizzazione_progress = {'percentuale': 0, 'log': [], 'completed': False}
reindicizzazione_lock = threading.Lock()

# P/ agendamento
schedule_lock = threading.Lock()
schedule_data = {"days": [], "hour": "00:00"}



def carica_pianificazione():
    global schedule_data
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            try:
                config = json.load(f)
                if 'schedule' in config:
                    schedule_data = config['schedule']
            except:
                pass

def salva_pianificazione(new_schedule):
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config_completo = json.load(f)
        except json.JSONDecodeError:
            config_completo = {}
    else:
        config_completo = {}
    config_completo['schedule'] = new_schedule
    with open(config_file, 'w') as f:
        json.dump(config_completo, f, ensure_ascii=False, indent=4)

carica_pianificazione()

# ==================== Decorator login_required ====================
from functools import wraps
def login_required(f):
    @wraps(f)
    def wrapper(*a, **k):
        if not session.get("logged_in"):
            return redirect(url_for("index"))
        return f(*a, **k)
    return wrapper


# ==================== ROTA PRINCIPAL (INDEX) ====================
@app.route('/')
def index():
    return render_template(
        'index.html',
        directories_indicizzate=directories_indicizzate,
        categorie=categorie
    )
# --- CONFIG PAGINA CONVERSORE AUTOMATICO (mostrar página) ---
@app.route("/configurazione_conversor")
@login_required
def configurazione_conversor():
    cfg = load_config()
    regole = cfg.get("conversion_watch", [])
    dirs_index = cfg.get("directories_indicizzate", directories_indicizzate)  # fallback seguro

    with auto_conv_lock:
        logs = auto_conversion_status.get("log", [])[-60:]
        stats = auto_conversion_status.get("stats", {})
        last_scan = stats.get("last_scan", "-")

    return render_template(
        "configurazione_conversor.html",
        config_watch=regole,
        directories_indicizzate=dirs_index,
        auto_logs=logs,
        monitor_on=True,
        last_scan=last_scan
    )

# --- SALVA / ATUALIZA REGRA (UM BOTÃO) ---
@app.route("/salva_config_conversor", methods=["POST"])
@login_required
def salva_config_conversor():
    input_dir  = (request.form.get("input_dir") or "").strip()
    output_dir = (request.form.get("output_dir") or "").strip()

    # Opções ligadas à regra
    def _to_int(v, default=None):
        try:
            return int(v) if (v is not None and str(v).strip() != "") else default
        except Exception:
            return default

    width       = _to_int(request.form.get("width"))
    height      = _to_int(request.form.get("height"))
    quality     = _to_int(request.form.get("quality"), 90)
    workers     = _to_int(request.form.get("workers"), 1)
    square      = bool(request.form.get("square"))
    colorize_ia = bool(request.form.get("colorize_ia"))
    keep_aspect = bool(request.form.get("keep_aspect"))

    if not input_dir or not output_dir:
        flash("Specificare input e output.", "error")
        return redirect(url_for("configurazione_conversor"))

    cfg = load_config()
    regole = cfg.get("conversion_watch", [])

    # Regra com opzioni annesse
    new_rule = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "options": {
            "width": width,
            "height": height,
            "quality": quality,
            "workers": workers,
            "square": square,
            "keep_aspect": keep_aspect,
            "colorize_ia": colorize_ia,
            "output_format": "jpg"
        }
    }

    # Se já existir (mesmo input+output), ATUALIZA opções; senão, adiciona
    updated = False
    for r in regole:
        if r.get("input_dir") == input_dir and r.get("output_dir") == output_dir:
            r["options"] = new_rule["options"]
            updated = True
            break
    if not updated:
        regole.append(new_rule)

    cfg["conversion_watch"] = regole
    save_config(cfg)

    _auto_log(f"[CONFIG] {'Aggiornata' if updated else 'Aggiunta'} regola: "
              f"IN='{input_dir}' → OUT='{output_dir}' "
              f"opts={new_rule['options']}")

    flash("Regola salvata con successo.", "success")
    return redirect(url_for("configurazione_conversor"))


@app.route("/rimuovi_config_conversor", methods=["POST"])
@login_required
def rimuovi_config_conversor():
    input_dir = request.form.get("input_dir")
    output_dir = request.form.get("output_dir")
    cfg = load_config()
    regole = cfg.get("conversion_watch", [])
    new_regole = [r for r in regole if not (r.get("input_dir")==input_dir and r.get("output_dir")==output_dir)]
    cfg["conversion_watch"] = new_regole
    save_config(cfg)
    flash("Regola rimossa.", "info")
    _auto_log(f"[CONFIG] Rimossa regola: IN='{input_dir}' → OUT='{output_dir}'")
    return redirect(url_for("configurazione_conversor"))


# ==================== LOGIN ====================
@app.route('/fazer_login', methods=['POST'])
def fazer_login():
    username = request.form.get('username')
    password = request.form.get('password')
    users = carica_utenti()
    for user in users:
        if user['username'] == username and user['password'] == password:
            session['logged_in'] = True
            session['username'] = username
            flash('Accesso effettuato con successo!', 'success')
            return redirect(url_for('index'))
    flash('Nome utente o password non validi!', 'error')
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.clear()
    flash('Logout effettuato.', 'info')
    return redirect(url_for('index'))
#  app.py  – logo depois das outras rotas -----------------------------


# ==================== PÁGINA DE ABOUT ====================

@app.route('/about')
@login_required
def about():
    return render_template('about.html')

# ==================== PÁGINA CONFIG (Impostazioni) ====================
@app.route('/configurazioni')
@login_required
def configurazioni():
    return render_template('config.html', categorie=categorie)

@app.route('/utilidades')
@login_required
def utilidades():
    return render_template('utilidades.html')

# ==================== NOVA PÁGINA DE INDICIZZAZIONE ====================
@app.route('/indicizzazione')
@login_required
def indicizzazione():
    return render_template(
        'indicizzazione.html',
        directories_indicizzate=directories_indicizzate
    )

# ==================== FUNÇÕES E ROTAS DE INDEXAÇÃO ====================
from indicizza import indicizza_immagini
from modello import cerca_immagini, encontrar_duplicatas, extrai_features_imagem, cerca_per_embedding

@app.route('/indicizza', methods=['POST'])
@login_required
def indicizza():
    global indicizzazione_thread
    global directories_indicizzate, indicizzazione_interrompida

    directory_input = request.form.get('directory_indicizza')
    nome_input = request.form.get('nome_indicizza', '').strip()

    if not directory_input:
        flash('Errore: Il campo della directory è obbligatorio.', 'error config')
        return redirect(url_for('indicizzazione'))
    if not os.path.isdir(directory_input):
        flash('Errore: La directory fornita non è valida.', 'error config')
        return redirect(url_for('indicizzazione'))

    for d in directories_indicizzate:
        if d['path'] == directory_input:
            flash('La directory è già stata indicizzata.', 'info config')
            return redirect(url_for('indicizzazione'))

    if indicizzazione_thread and indicizzazione_thread.is_alive():
        flash('Indicizzazione già in corso.', 'error config')
        return redirect(url_for('indicizzazione'))

    indicizzazione_interrompida.clear()
    directories_indicizzate.append({"path": directory_input, "nome": nome_input})
    salva_configurazione(directories_indicizzate)

    with progress_lock:
        indicizzazione_progress['percentuale'] = 0
        indicizzazione_progress['log'] = []

    def run_indicizzazione():
        try:
            lista_paths = [d["path"] for d in directories_indicizzate]
            indicizza_immagini(lista_paths, file_indice_faiss, file_percorsi, indicizzazione_interrompida)
            with progress_lock:
                if indicizzazione_interrompida.is_set():
                    indicizzazione_progress['log'].append("Indicizzazione interrotta dall'utente.")
                else:
                    indicizzazione_progress['percentuale'] = 100
                    indicizzazione_progress['log'].append("Indicizzazione completata con successo.")
        except Exception as e:
            with progress_lock:
                indicizzazione_progress['log'].append(f"Errore durante l'indicizzazione: {e}")
            logger.error("Errore durante l'indicizzazione: %s", e)

    indicizzazione_thread = threading.Thread(target=run_indicizzazione)
    indicizzazione_thread.start()

    flash(f'Indicizzazione avviata per "{directory_input}".', 'success config')
    return redirect(url_for('indicizzazione'))

@app.route('/interrompi_indicizzazione', methods=['POST'])
@login_required
def interrompi_indicizzazione_route():
    logger.debug("Interrompendo indicizzazione...")
    indicizzazione_interrompida.set()
    flash('Indicizzazione interrotta.', 'info config')
    return redirect(url_for('indicizzazione'))

@app.route('/stato_indicizzazione')
@login_required
def stato_indicizzazione():
    with progress_lock:
        progress_data = indicizzazione_progress.copy()
    return jsonify(progress_data)

@app.route('/reindicizza_tutto', methods=['POST'])
@login_required
def reindicizza_tutto():
    global indicizzazione_thread
    global directories_indicizzate, indicizzazione_interrompida

    indicizzazione_interrompida.set()
    if indicizzazione_thread and indicizzazione_thread.is_alive():
        indicizzazione_thread.join()

    if os.path.exists('embeddings_immagini.npy'):
        os.remove('embeddings_immagini.npy')
    if os.path.exists('indice_faiss.index'):
        os.remove('indice_faiss.index')
    if os.path.exists('percorsi_immagini.pkl'):
        os.remove('percorsi_immagini.pkl')

    with progress_lock:
        indicizzazione_progress['percentuale'] = 0
        indicizzazione_progress['log'] = ["Reindicizzazione completa avviata..."]

    indicizzazione_interrompida.clear()

    def run_indicizzazione_completa():
        try:
            lista_paths = [d["path"] for d in directories_indicizzate]
            indicizza_immagini(lista_paths, file_indice_faiss, file_percorsi, indicizzazione_interrompida)
            with progress_lock:
                if indicizzazione_interrompida.is_set():
                    indicizzazione_progress['log'].append("Reindicizzazione interrotta dall'utente.")
                else:
                    indicizzazione_progress['percentuale'] = 100
                    indicizzazione_progress['log'].append("Reindicizzazione completata con successo.")
        except Exception as e:
            with progress_lock:
                indicizzazione_progress['log'].append(f"Errore durante la reindicizzazione: {e}")
            logger.error("Errore durante a reindicização: %s", e)

    indicizzazione_thread = threading.Thread(target=run_indicizzazione_completa)
    indicizzazione_thread.start()

    flash("Reindicizzazione completa avviata.", "success config")
    return redirect(url_for('indicizzazione'))

@app.route('/aggiorna_nome_directory', methods=['POST'])
@login_required
def aggiorna_nome_directory():
    path = request.form.get('path')
    display_name = request.form.get('display_name', '').strip()
    found = False
    for d in directories_indicizzate:
        if d["path"] == path:
            d["nome"] = display_name
            found = True
            break

    if found:
        salva_configurazione(directories_indicizzate)
        flash('Nome della directory aggiornato con successo.', 'success config')
    else:
        flash('Directory non trovata.', 'error config')
    return redirect(url_for('indicizzazione'))




def filtra_imagens(filtros):
    """
    Retorna lista de imagens que satisfaz TODOS os filtros ativos.
    filtros = {"tags": [...], "dirs": [...], "colors": [...]}
    """
    todas = carrega_todas_imagens()   # implementa: lista completa

    result = []
    for img in todas:
        ok = True
        if filtros["tags"]:
            ok = ok and any(t in img.tags for t in filtros["tags"])
        if filtros["dirs"]:
            ok = ok and any(img.path.startswith(d) for d in filtros["dirs"])
        if filtros["colors"]:
            ok = ok and img.color in filtros["colors"]
        if ok:
            result.append(img.path)
    return result

@app.route('/rimuovi_directory', methods=['POST'])
@login_required
def rimuovi_directory():
    global directories_indicizzate
    directory_da_rimuovere = request.form.get('directory_da_rimuovere')
    found_index = None
    for i, d in enumerate(directories_indicizzate):
        if d["path"] == directory_da_rimuovere:
            found_index = i
            break

    if found_index is not None:
        directories_indicizzate.pop(found_index)
        salva_configurazione(directories_indicizzate)
        reindicizza_dopo_rimozione()
        flash(f'Directory "{directory_da_rimuovere}" rimossa.', 'success config')
    else:
        flash('La directory non è presente nella lista.', 'error config')
    return redirect(url_for('indicizzazione'))

# Chamado após remover directories, para “reindex incremental”
def reindicizza_dopo_rimozione():
    def run_reindicizzazione():
        with reindicizzazione_lock:
            reindicizzazione_progress['percentuale'] = 0
            reindicizzazione_progress['log'] = []
            reindicizzazione_progress['completed'] = False
            reindicizzazione_progress['log'].append("Inizio della reindicizzazione...")

        try:
            if os.path.exists('embeddings_immagini.npy') and os.path.exists(file_percorsi):
                with open('embeddings_immagini.npy', 'rb') as f:
                    embeddings = np.load(f)
                with open(file_percorsi, 'rb') as f:
                    percorsi_immagini = pickle.load(f)

                nuovi_percorsi = []
                nuovi_embeddings = []
                total_images = len(percorsi_immagini)
                lista_paths = [d["path"] for d in directories_indicizzate]

                for idx, percorso in enumerate(percorsi_immagini):
                    if os.path.exists(percorso) and any(percorso.startswith(dir_) for dir_ in lista_paths):
                        nuovi_percorsi.append(percorso)
                        nuovi_embeddings.append(embeddings[idx])
                    with reindicizzazione_lock:
                        percentuale = int((idx / total_images) * 100)
                        reindicizzazione_progress['percentuale'] = percentuale
                        if idx % 10 == 0 or idx == total_images - 1:
                            reindicizzazione_progress['log'].append(
                                f"Processate {idx + 1} di {total_images} immagini."
                            )
                if nuovi_embeddings:
                    embedding_array = np.vstack(nuovi_embeddings).astype('float32')
                    faiss.normalize_L2(embedding_array)

                    with open('embeddings_immagini.npy', 'wb') as f:
                        np.save(f, embedding_array)

                    dimensione = embedding_array.shape[1]
                    indice_faiss_ = faiss.IndexFlatIP(dimensione)
                    indice_faiss_.add(embedding_array)
                    faiss.write_index(indice_faiss_, file_indice_faiss)

                    with open(file_percorsi, 'wb') as f:
                        pickle.dump(nuovi_percorsi, f)

                    with reindicizzazione_lock:
                        reindicizzazione_progress['percentuale'] = 100
                        reindicizzazione_progress['completed'] = True
                        reindicizzazione_progress['log'].append("Reindicizzazione completata con successo.")
                else:
                    if os.path.exists('embeddings_immagini.npy'):
                        os.remove('embeddings_immagini.npy')
                    if os.path.exists('indice_faiss.index'):
                        os.remove('indice_faiss.index')
                    if os.path.exists('percorsi_immagini.pkl'):
                        os.remove('percorsi_immagini.pkl')
                    with reindicizzazione_lock:
                        reindicizzazione_progress['percentuale'] = 100
                        reindicizzazione_progress['completed'] = True
                        reindicizzazione_progress['log'].append("Nessuna immagine da indicizzare. Dati rimossi.")
            else:
                with reindicizzazione_lock:
                    reindicizzazione_progress['completed'] = True
                    reindicizzazione_progress['log'].append("Nessun indice da aggiornare.")
        except Exception as e:
            with reindicizzazione_lock:
                reindicizzazione_progress['log'].append(f"Errore durante la reindicizzazione: {e}")
                reindicizzazione_progress['completed'] = True

    reindicizzazione_thread = threading.Thread(target=run_reindicizzazione)
    reindicizzazione_thread.start()


@app.route('/stato_reindicizzazione_page')
@login_required
def stato_reindicizzazione_page():
    return render_template('stato_reindicizzazione.html')

@app.route('/stato_reindicizzazione')
@login_required
def stato_reindicizzazione():
    with reindicizzazione_lock:
        progress_data = reindicizzazione_progress.copy()
    return jsonify(progress_data)

# ==================== ROTA DE PESQUISA (texto ou imagem) ====================
@app.route("/ricerca", methods=["POST"])
@login_required
def ricerca():
    # diretórios selecionados no formulário (ou mantém os que já estão na sessão)
    dirs_sel = request.form.getlist("directories_selezionate") or session.get("directories_selezionate") or [
        d["path"] if isinstance(d, dict) else d for d in directories_indicizzate
    ]
    # persiste imediatamente (qualquer fluxo depois usará isto)
    session["directories_selezionate"] = dirs_sel

    # ========================= PESQUISA POR IMAGEM =========================
    image_file = request.files.get("image_file")
    if image_file and image_file.filename:
        import tempfile
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image_file.save(tmp.name)
        try:
            emb = extrai_features_imagem(tmp.name)
            risultati, totali = cerca_per_embedding(
                emb, file_indice_faiss, file_percorsi,
                directories_selezionate=dirs_sel,
                offset=0, limit=100
            )
        finally:
            tmp.close()
            os.remove(tmp.name)

        if not totali:
            flash("Nessuna immagine simile trovata.", "info index")
            return redirect(url_for("index"))

        # estatística simples
        stats = Counter(_load_stats())
        for r in risultati:
            stats[r["percorso"]] += 1
        _save_stats(stats)

        # limpa contexto de texto/tags, mas mantém diretórios na sessão
        session.update({
            "termo_busca": "", "categoria": "",
            "tags_selezionate": "",
            FILTERS_KEY: {}
        })
        return render_template("risultati.html",
                               immagini=risultati,
                               total_resultados=totali,
                               per_page=100,
                               categorie=categorie,
                               directories_indicizzate=directories_indicizzate)

    # ========================= PESQUISA POR TEXTO =========================
    termo      = (request.form.get("input_testo") or "").strip()
    categoria  = (request.form.get("categoria") or "").strip()
    tags_raw   = request.form.get("tags_selezionate", "")
    tags_list  = [t for t in tags_raw.split(",") if t]

    # monta descrição base
    descr_base = " ".join([termo] + tags_list).strip()
    if not descr_base and categoria:
        descr_base = categoria.strip()

    # ========================= FEED (sem termo/tag) → **RESPEITA dirs** =========================
    if not descr_base:
        if not os.path.isfile(file_percorsi):
            flash("Nessuna immagine indicizzata.", "info index")
            return redirect(url_for("index"))

        percorsi = pickle.load(open(file_percorsi, "rb"))
        # AQUI: filtra pelos diretórios selecionados SEMPRE
        percorsi = _filter_paths_by_dirs(percorsi, dirs_sel)

        stats = _load_stats()
        percorsi.sort(key=lambda p: (-stats.get(p, 0), p))

        top = percorsi[:100] if len(percorsi) > 0 else []
        immagini = [
            {"percorso": p, "percorso_url": p, "extra_ocr_match": False}
            for p in top
        ]
        totali = len(percorsi)

        session.update({
            "termo_busca": "",
            "categoria": "",
            "tags_selezionate": "",
            FILTERS_KEY: {}
        })
        return render_template(
            "risultati.html",
            immagini=immagini,
            total_resultados=totali,
            per_page=100,
            categorie=categorie,
            directories_indicizzate=directories_indicizzate
        )

    # ========================= COM DESCRIÇÃO → OCR-first (já respeita dirs internamente) =========================
    immagini, totali = cerca_immagini(
        descrizione=descr_base,
        categoria=categoria,
        file_indice_faiss=file_indice_faiss,
        file_percorsi=file_percorsi,
        directories_selezionate=dirs_sel,
        offset=0, limit=100,
        prefer_ocr=True
    )

    if immagini:
        stats = Counter(_load_stats())
        for r in immagini:
            stats[r["percorso"]] += 1
        _save_stats(stats)

    session.update({
        "termo_busca": "",
        "categoria": "",
        "tags_selezionate": "",
        FILTERS_KEY: {}
    })

    return render_template(
        "risultati.html",
        immagini=immagini,
        total_resultados=totali,
        per_page=100,
        categorie=categorie,
        directories_indicizzate=directories_indicizzate
    )

# -----------------------------------------------------------------
@app.route("/carregar_mais_imagens", methods=["POST"])
@login_required
def carregar_mais_imagens():
    data = request.get_json() or {}
    cur  = int(data.get("current_index", 0))
    step = int(data.get("per_page", 100))

    extra      = session.get(FILTERS_KEY, {})
    termo_base = session.get("termo_busca", "")
    termo_extra= extra.get("text", "")
    termo      = " ".join(filter(None, [termo_base, termo_extra])).strip()

    categoria  = extra.get("categoria") or session.get("categoria", "")
    tags       = [t for t in session.get("tags_selezionate","").split(",") if t] + extra.get("tags", [])
    colori     = extra.get("color", [])
    dirs_sel   = session.get("directories_selezionate") or [
        d["path"] if isinstance(d, dict) else d for d in directories_indicizzate
    ]
    # se o filtro extra trouxe dirs, use-o e persista
    if extra.get("dirs"):
        dirs_sel = extra.get("dirs")
        session["directories_selezionate"] = dirs_sel

    # ========================= FEED POPULAR (sem termo/filters) → **RESPEITA dirs** =========================
    if not any([termo, categoria, tags, colori]):
        if not os.path.isfile(file_percorsi):
            return jsonify({"imagens": [], "end": True})
        percorsi = pickle.load(open(file_percorsi, "rb"))
        percorsi = _filter_paths_by_dirs(percorsi, dirs_sel)   # <<<<<< filtro de diretórios

        stats  = _load_stats()
        percorsi.sort(key=lambda p: (-stats.get(p,0), p))
        slice_   = percorsi[cur:cur+step]
        end_flag = (cur+len(slice_)) >= len(percorsi)
        imgs     = [{"percorso": p, "percorso_url": p, "extra_ocr_match": False} for p in slice_]
        return jsonify({"imagens": imgs, "end": end_flag})

    # ========================= COM FILTROS → OCR-first (model já filtra dirs) =========================
    descr_full = " ".join(filter(None, [termo] + tags + colori))
    imgs, tot = cerca_immagini(
        descrizione = descr_full,
        categoria   = categoria,
        file_indice_faiss = file_indice_faiss,
        file_percorsi     = file_percorsi,
        directories_selezionate = dirs_sel,
        offset = cur, limit = step,
        prefer_ocr = True
    )
    return jsonify({"imagens": imgs, "end": (cur+len(imgs))>=tot})


# -----------------------  /galleria  ------------------------------------
@app.route("/galleria")
@login_required
def galleria():
    PER_PAGE = 100

    # diretórios em vigor: o que veio da sessão (setado em /ricerca ou pelo filtro da UI)
    dirs_sel = session.get("directories_selezionate") or [
        d["path"] if isinstance(d, dict) else d
        for d in directories_indicizzate
    ]

    extra = session.get(FILTERS_KEY, {})   # {text,categoria,tags,color,dirs}

    termo_f   = (extra.get("text") or "").strip()
    cat_f     = (extra.get("categoria") or "").strip()
    tags_f    = extra.get("tags", []) or []
    colori_f  = extra.get("color", []) or []
    dirs_f    = extra.get("dirs", []) or []

    # Se o usuário clicou em diretório na UI de risultati, ele manda em tudo
    if dirs_f:
        dirs_sel = dirs_f
        session["directories_selezionate"] = dirs_sel  # persiste

    ha_filtro = any([termo_f, cat_f, tags_f, colori_f])

    # ============== 1) FEED POPULAR (sem filtros) – **RESPEITA dirs_sel** ==========
    if not ha_filtro:
        percorsi = pickle.load(open(file_percorsi, "rb")) if os.path.isfile(file_percorsi) else []
        # filtro por diretórios SEMPRE
        percorsi = _filter_paths_by_dirs(percorsi, dirs_sel)

        stats = _load_stats()
        percorsi.sort(key=lambda p: (-stats.get(p, 0), p))

        top = percorsi[:PER_PAGE] or (random.sample(percorsi, min(PER_PAGE, len(percorsi))) if percorsi else [])
        immagini = [{"percorso": p, "percorso_url": p, "extra_ocr_match": False} for p in top]
        tot = len(percorsi)

    # ============== 2) COM FILTROS → OCR-first (model já filtra dirs) ==============
    else:
        descr = " ".join(filter(None, [termo_f] + tags_f + colori_f))
        categoria = cat_f

        immagini, tot = cerca_immagini(
            descrizione             = descr,
            categoria               = categoria,
            file_indice_faiss       = file_indice_faiss,
            file_percorsi           = file_percorsi,
            directories_selezionate = dirs_sel,
            offset=0, limit=PER_PAGE,
            prefer_ocr=True
        )

        # estatística simples do “feed” filtrado
        if immagini:
            stats = Counter(_load_stats())
            for r in immagini:
                stats[r["percorso"]] += 1
            _save_stats(stats)

    return render_template(
        "risultati.html",
        immagini                = immagini,
        total_resultados        = tot,
        per_page                = PER_PAGE,
        categorie               = categorie,
        directories_indicizzate = directories_indicizzate
    )

@app.route("/set_filtros_galeria", methods=["POST"])
@login_required
def set_filtros_galeria():
    """
    Recebe JSON, grava no cookie de sessão e — se presente — atualiza
    também a lista de diretórios activos para as próximas chamadas.
      payload = {
          text: "...", categoria:"...", tags:[...], color:[...], dirs:[...]
      }
    """
    payload = request.get_json(force=True) or {}

    # guarda tudo em FILTERS_KEY  (text, categoria, tags, color, dirs)
    session[FILTERS_KEY] = payload

    # se o usuário escolheu diretórios, actualizamos o contexto
    if payload.get("dirs"):                   # lista de paths
        session["directories_selezionate"] = payload["dirs"]

    return ("", 204)   # resposta vazia / sucesso



# --- logo depois da rota /set_filtros_galeria ----------------------------
@app.route("/ricerca_simili", methods=["POST"])
@login_required
def ricerca_simili():
    data     = request.get_json(force=True)
    img_path = data.get("path")
    dirs_sel = data.get("dirs") or [
        d["path"] if isinstance(d, dict) else d for d in directories_indicizzate
    ]

    if not (img_path and os.path.exists(img_path)):
        flash("Immagine non trovata.", "error")
        return redirect(url_for("index"))

    # extrai embedding da própria imagem
    try:
        emb = extrai_features_imagem(img_path)
    except Exception as e:
        flash(f"Errore nell'elaborazione dell'immagine: {e}", "error")
        return redirect(url_for("index"))

    risultati, tot = cerca_per_embedding(
        emb, file_indice_faiss, file_percorsi,
        directories_selezionate=dirs_sel, offset=0, limit=100
    )

    # estatísticas “popularità”
    stats = Counter(_load_stats())
    for r in risultati:
        stats[r["percorso"]] += 1
    _save_stats(stats)

    # mantém contexto na sessão
    session.update({
        "termo_busca": "", "categoria": "",
        "tags_selezionate": "", "directories_selezionate": dirs_sel,
        FILTERS_KEY: {}
    })

    return render_template("risultati.html",
                           immagini=risultati,
                           total_resultados=tot,
                           per_page=100,
                           categorie=categorie,
                           directories_indicizzate=directories_indicizzate)

@app.route('/immagini/<path:percorso>')
@login_required
def immagini(percorso):
    """Serve o arquivo original exatamente onde está no disco (bind mount).
       Se vier sem '/', tornamos absoluto para evitar 404 por caminho relativo."""
    from urllib.parse import unquote
    safe_path = os.path.normpath(unquote(percorso))

    # endurece: se não for absoluto, prefixa '/'
    if not os.path.isabs(safe_path):
        safe_path = "/" + safe_path

    if os.path.exists(safe_path):
        return send_file(safe_path)
    return "Immagine non trovata", 404


@app.route('/miniatura/<path:percorso>')
@login_required
def miniatura(percorso):
    """Gera e devolve miniatura 200×200 em JPEG (em memória).
       Se vier sem '/', tornamos absoluto para evitar 404 por caminho relativo."""
    from urllib.parse import unquote
    safe_path = os.path.normpath(unquote(percorso))
    if not os.path.isabs(safe_path):
        safe_path = "/" + safe_path

    if not os.path.exists(safe_path):
        return "Immagine non trovata", 404

    with Image.open(safe_path) as img:
        img.thumbnail((200, 200))
        buf = BytesIO()
        img.save(buf, 'JPEG')
        buf.seek(0)
        return send_file(buf, mimetype='image/jpeg')


@app.route('/get_metadata', methods=['POST'])
@login_required
def get_metadata():
    data = request.get_json()
    image_path = data.get('image_path')
    if not image_path:
        return jsonify({'error': 'Percorso immagine non fornito'}), 400
    if not os.path.exists(image_path):
        return jsonify({'error': 'Immagine non trovata'}), 404
    try:
        result = subprocess.run(
            ['exiftool', '-j', '-XMP:All', image_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            metadati = {'Errore': result.stderr}
        else:
            metadata_json = json.loads(result.stdout)
            if metadata_json:
                metadati_tutti = metadata_json[0]
                metadati_filtrati = {
                    'dc:subject': metadati_tutti.get('Subject', ''),
                    'dc:title': metadati_tutti.get('Title', ''),
                    'xmp:rating': metadati_tutti.get('Rating', '')
                }
                metadati = metadati_filtrati
            else:
                metadati = {'Metadati': 'Non disponibili.'}
    except Exception as e:
        metadati = {'Errore': str(e)}
        logger.error("get_metadata erro: %s", e)
    return jsonify({'metadati': metadati})

# ==================== DUPLICATAS ====================
@app.route('/encontrar_duplicatas', methods=['GET', 'POST'])
@login_required
def encontrar_duplicatas_route():
    global directories_indicizzate, embeddings_file, file_percorsi, encontrar_progress

    if not directories_indicizzate:
        flash('Errore: Devi indicizzare prima le immagini.', 'error duplicates')
        return redirect(url_for('utilidades'))

    if request.method == 'POST':
        similaridade_input = request.form.get('similaridade')
        try:
            similaridade_input = similaridade_input.replace(',', '.')
            similaridade_valor = float(similaridade_input)
            if not 0 <= similaridade_valor <= 1:
                raise ValueError("Il valore deve essere tra 0 e 1.")
        except ValueError as ve:
            flash(f'Errore: {ve}', 'error duplicates')
            return redirect(url_for('encontrar_duplicatas_route'))
        except Exception:
            flash('Errore: Inserisci un valore di similarità valido.', 'error duplicates')
            return redirect(url_for('encontrar_duplicatas_route'))
    else:
        similaridade_valor = 0.99

    session['similaridade_valor'] = similaridade_valor

    with encontrar_lock:
        encontrar_progress['percentuale'] = 0
        encontrar_progress['log'] = []
        encontrar_progress['completed'] = False
        encontrar_progress['grupos_duplicatas'] = None
        encontrar_progress['similaridade_valor'] = similaridade_valor
        encontrar_progress['total_duplicatas'] = 0

    def run_encontrar_duplicatas():
        try:
            with encontrar_lock:
                encontrar_progress['log'].append("Inizio della ricerca di duplicati...")
                encontrar_progress['percentuale'] = 5

            grupos_duplicatas = encontrar_duplicatas(
                embeddings_file,
                file_percorsi,
                threshold_duplicates=similaridade_valor
            )
            with encontrar_lock:
                encontrar_progress['log'].append("Ricerca completata. Elaborazione dei risultati...")
                encontrar_progress['percentuale'] = 80

            total_duplicatas = sum(len(g) - 1 for g in grupos_duplicatas if len(g) > 1)

            with encontrar_lock:
                encontrar_progress['grupos_duplicatas'] = grupos_duplicatas
                encontrar_progress['total_duplicatas'] = total_duplicatas
                encontrar_progress['percentuale'] = 100
                encontrar_progress['completed'] = True

                if total_duplicatas == 0:
                    encontrar_progress['log'].append("Nessuna immagine duplicata trovata.")
                else:
                    encontrar_progress['log'].append(f"Trovate {total_duplicatas} immagini duplicate.")
                encontrar_progress['log'].append("Ricerca duplicati completata con successo.")

        except Exception as e:
            with encontrar_lock:
                encontrar_progress['log'].append(f"Errore durante la ricerca di duplicati: {e}")
                encontrar_progress['percentuale'] = 100
                encontrar_progress['completed'] = True
            logger.error("Errore busca duplicatas: %s", e)

    t = threading.Thread(target=run_encontrar_duplicatas)
    t.start()
    return redirect(url_for('stato_encontrar_page'))

@app.route('/stato_encontrar_page')
@login_required
def stato_encontrar_page():
    return render_template('stato_encontrar.html')

@app.route('/stato_encontrar')
@login_required
def stato_encontrar():
    with encontrar_lock:
        progress_data = {
            'percentuale': encontrar_progress['percentuale'],
            'log': encontrar_progress['log'],
            'completed': encontrar_progress['completed']
        }
    return jsonify(progress_data)

@app.route('/mostrar_duplicatas')
@login_required
def mostrar_duplicatas():
    global encontrar_progress
    with encontrar_lock:
        if not encontrar_progress['completed']:
            return redirect(url_for('stato_encontrar_page'))
        grupos_duplicatas = encontrar_progress.get('grupos_duplicatas', [])
        total_duplicatas = encontrar_progress.get('total_duplicatas', 0)
        similaridade_valor = encontrar_progress.get('similaridade_valor', 0.95)

    if total_duplicatas == 0 or not grupos_duplicatas:
        flash('Nessuna immagine duplicata trovata.', 'info duplicates')
        return render_template('duplicatas.html', grupos=[], total_duplicatas=0, similaridade=similaridade_valor)
    else:
        return render_template('duplicatas.html',
                               grupos=grupos_duplicatas,
                               total_duplicatas=total_duplicatas,
                               similaridade=similaridade_valor)

@app.route('/eliminar_duplicatas')
@login_required
def eliminar_duplicatas_route():
    global encontrar_progress, eliminazione_progress
    with eliminazione_lock:
        eliminazione_progress['percentuale'] = 0
        eliminazione_progress['log'] = []
        eliminazione_progress['completed'] = False

    with encontrar_lock:
        grupos_duplicatas = encontrar_progress.get('grupos_duplicatas', [])
        total_duplicatas = encontrar_progress.get('total_duplicatas', 0)

    if not grupos_duplicatas or total_duplicatas == 0:
        flash('Nessuna immagine duplicata trovata per l\'eliminazione.', 'info duplicates')
        return redirect(url_for('utilidades'))

    def run_eliminazione_duplicatas():
        global directories_indicizzate
        try:
            total_grupos = len(grupos_duplicatas)
            processed_grupos = 0
            lista_paths = [d["path"] for d in directories_indicizzate]

            for grupo in grupos_duplicatas:
                with eliminazione_lock:
                    percentuale = int((processed_grupos / total_grupos) * 100)
                    eliminazione_progress['percentuale'] = percentuale
                    eliminazione_progress['log'].append(
                        f"Eliminando duplicati del gruppo {processed_grupos + 1} di {total_grupos}."
                    )

                grupo_sorted = sorted(grupo, key=lambda x: x['data_modifica'], reverse=True)
                images_to_delete = grupo_sorted[1:]
                for img in images_to_delete:
                    image_path = img['percorso']
                    if os.path.exists(image_path):
                        if any(image_path.startswith(p) for p in lista_paths):
                            os.remove(image_path)
                            with eliminazione_lock:
                                eliminazione_progress['log'].append(f"Immagine eliminata: {image_path}")
                        else:
                            with eliminazione_lock:
                                eliminazione_progress['log'].append(
                                    f"Immagine non eliminata (non in directory indicizzata): {image_path}"
                                )
                    else:
                        with eliminazione_lock:
                            eliminazione_progress['log'].append(f"Immagine non trovata: {image_path}")
                processed_grupos += 1

            with eliminazione_lock:
                eliminazione_progress['percentuale'] = 100
                eliminazione_progress['completed'] = True
                eliminazione_progress['log'].append("Eliminazione completata con successo.")
            reindicizza_dopo_rimozione()
        except Exception as e:
            with eliminazione_lock:
                eliminazione_progress['log'].append(f"Errore durante l'eliminazione: {e}")
                eliminazione_progress['completed'] = True
            logger.error("Errore elim duplicatas: %s", e)

    eliminazione_thread = threading.Thread(target=run_eliminazione_duplicatas)
    eliminazione_thread.start()
    return redirect(url_for('stato_eliminazione_page'))

@app.route('/stato_eliminazione_page')
@login_required
def stato_eliminazione_page():
    return render_template('stato_eliminazione.html')

@app.route('/stato_eliminazione')
@login_required
def stato_eliminazione():
    with eliminazione_lock:
        progress_data = eliminazione_progress.copy()
    return jsonify(progress_data)

# CONVERSOR
@app.route("/conversor", methods=["GET", "POST"])
@login_required
def conversor():
    global conversion_status

    if request.method == "POST":
        dir_path         = request.form.get("percorso_input", "")
        width            = request.form.get("width")   or None
        height           = request.form.get("height")  or None
        workers          = int(request.form.get("workers") or os.cpu_count())
        skip_duplicates  = bool(request.form.get("skip_duplicates"))
        colorize_ia      = bool(request.form.get("colorize_ia"))

        width  = int(width)  if width  else None
        height = int(height) if height else None

        icc_profile_path = None
        icc_file = request.files.get("icc_profile")
        if icc_file and icc_file.filename:
            tmpdir = tempfile.gettempdir()
            icc_profile_path = os.path.join(
                tmpdir, secure_filename(icc_file.filename)
            )
            icc_file.save(icc_profile_path)

        from conversor import ConversionOptions, bulk_convert  # import local p/ evitar custo no import global
        opts = ConversionOptions(
            width=width,
            height=height,
            square=False,  # default
            keep_aspect=True,  # default
            skip_duplicates=True,
            icc_profile=icc_profile_path,
            workers=workers,
            output_format="jpg",  # forza JPEG
        )



        # ---------- 3. Esegui conversione ----------
        def _progress_cb(val):
            nonlocal progress
            progress = val                 # usato dal template

        logs, stats = bulk_convert(dir_path, opts, _progress_cb)

        flash(
            f"Conversione terminata – "
            f"{stats['converted']} ok, {stats['skipped']} saltati, "
            f"{stats['error']} errori.", "success"
        )
        progress = 100

        with conv_lock:
            conversion_status = {
                "log": ["🟢 Conversione avviata…"],
                "progress": 0,
                "running": True,
                "sent": 0
            }

        def _run():
            try:
                def _cb(pct):
                    with conv_lock:
                        conversion_status["progress"] = pct
                logs, stats = bulk_convert(dir_path, opts, _cb)
                with conv_lock:
                    conversion_status["log"].extend(logs)
                    conversion_status["log"].append(
                        f"=== Fine – OK {stats.get('converted',0)} | "
                        f"Saltati {stats.get('skipped',0)} | Errori {stats.get('error',0)}"
                    )
            except Exception as e:
                with conv_lock:
                    conversion_status["log"].append(f"❌ Errore: {e}")
            finally:
                with conv_lock:
                    conversion_status["progress"] = 100
                    conversion_status["running"]  = False

        threading.Thread(target=_run, daemon=True).start()
        flash("Conversione avviata – controlla il log in tempo reale.", "info")
        return redirect(url_for("conversor"))

    with conv_lock:
        logs_copy = list(conversion_status["log"])
        prog      = conversion_status["progress"]
    return render_template("conversor.html", logs=logs_copy, progress=prog, cpu_count=os.cpu_count())

@app.route("/stato_conversor")
@login_required
def stato_conversor():
    with conv_lock:
        sent = conversion_status["sent"]
        new  = conversion_status["log"][sent:]
        conversion_status["sent"] = sent + len(new)
        return jsonify({
            "progress": conversion_status["progress"],
            "running" : conversion_status["running"],
            "log"     : new
        })

# INFO IMMAGINI
@app.route('/info_immagini')
@login_required
def info_immagini():
    data = carica_info_immagini()
    return jsonify(data)

@app.route('/salva_info_immagine', methods=['POST'])
@login_required
def salva_info_immagine():
    try:
        payload = request.get_json()
        percorso = payload.get('percorso')
        if not percorso:
            return jsonify({'status': 'error', 'msg': 'Percorso non fornito'}), 400

        esclusiva = bool(payload.get('esclusiva'))
        scadenza = payload.get('scadenza', '')
        note = payload.get('note', '')

        info_data = carica_info_immagini()
        info_data[percorso] = {
            'esclusiva': esclusiva,
            'scadenza': scadenza,
            'note': note
        }
        salva_info_immagini(info_data)
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'status': 'error', 'msg': str(e)})

# ==================== PARAMETRI (configura_parametri) ====================
@app.route('/configura_parametri', methods=['GET'])
@login_required
def configura_parametri():
    files_data = [
        {"nome": "config.json",             "desc": "Configurazione Principale (config.json)"},
        {"nome": "categorie.json",          "desc": "Categorie e Tag (categorie.json)"},
        {"nome": "info_immagini.json",      "desc": "Metadati Immagini (info_immagini.json)"},
        {"nome": "embeddings_immagini.npy", "desc": "Embeddings delle Immagini (embeddings_immagini.npy)"},
        {"nome": "indice_faiss.index",      "desc": "Indice FAISS (indice_faiss.index)"},
        {"nome": "percorsi_immagini.pkl",   "desc": "Percorsi delle Immagini (percorsi_immagini.pkl)"},
        {"nome": "ocr_metadata.json","desc": "Metadati OCR (ocr_metadata.json)"},
        {"nome": "search_stats.json",       "desc": "Statistiche di Ricerca (search_stats.json)"},
    ]
    return render_template('parametri.html', files_data=files_data)

@app.route('/export_parametro', methods=['GET'])
@login_required
def export_parametro():
    file_name = request.args.get('file_name', '')
    if not file_name:
        flash('Nessun file specificato per l\'export.', 'error param')
        return redirect(url_for('configura_parametri'))

    full_path = os.path.join(os.getcwd(), file_name)
    if not os.path.exists(full_path):
        flash(f'File {file_name} non trovato sul server.', 'error param')
        return redirect(url_for('configura_parametri'))

    return send_file(full_path, as_attachment=True, download_name=file_name)

@app.route('/import_parametro', methods=['POST'])
@login_required
def import_parametro():
    file_name = request.form.get('file_name', '')
    if not file_name:
        flash('Nessun file specificato per l\'import.', 'error param')
        return redirect(url_for('configura_parametri'))

    if 'file_upload' not in request.files:
        flash('Nessun file inviato.', 'error param')
        return redirect(url_for('configura_parametri'))

    uploaded_file = request.files['file_upload']
    if uploaded_file.filename == '':
        flash('Nessun file selezionato.', 'error param')
        return redirect(url_for('configura_parametri'))

    full_path = os.path.join(os.getcwd(), file_name)
    try:
        uploaded_file.save(full_path)
        flash(f'File {file_name} importato con successo.', 'success param')
    except Exception as e:
        flash(f'Errore nell\'importazione di {file_name}: {e}', 'error param')

    return redirect(url_for('configura_parametri'))

@app.route('/recover_parametro', methods=['POST'])
@login_required
def recover_parametro():
    file_name = request.form.get('file_name', '')
    if not file_name:
        flash('Nessun file specificato.', 'error param')
        return redirect(url_for('configura_parametri'))

    full_path = os.path.join(os.getcwd(), file_name)
    old_path = full_path + '.old'
    if not os.path.exists(old_path):
        flash(f"Non esiste un file .old per {file_name}.", 'error param')
        return redirect(url_for('configura_parametri'))

    if os.path.exists(full_path):
        os.remove(full_path)
    os.rename(old_path, full_path)
    flash(f"Recuperata la versione precedente di {file_name}.", 'success param')
    return redirect(url_for('configura_parametri'))

@app.route('/export_tutti', methods=['GET'])
@login_required
def export_tutti():
    files_list = [
        'config.json',
        'categorie.json',
        'info_immagini.json',
        'embeddings_immagini.npy',
        'indice_faiss.index',
        'percorsi_immagini.pkl',
        'static/ocr_metadata.json',
        'search_stats.json'
    ]


    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    zip_path = temp.name
    temp.close()

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for f in files_list:
            if os.path.exists(f):
                zipf.write(f, arcname=os.path.basename(f))

    return send_file(zip_path, as_attachment=True, download_name="export_parametri.zip")

@app.route('/import_tutti', methods=['POST'])
@login_required
def import_tutti():
    if 'file_upload_zip' not in request.files:
        flash("Nessun file ZIP inviato.", 'error param')
        return redirect(url_for('configura_parametri'))

    upload = request.files['file_upload_zip']
    if upload.filename == '':
        flash("Nessun file selezionato.", 'error param')
        return redirect(url_for('configura_parametri'))

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    zip_path = temp.name
    temp.close()

    upload.save(zip_path)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zipf.extractall(path=os.getcwd())
        flash("Importazione in blocco completata con successo.", 'success param')
    except Exception as e:
        flash(f"Errore durante l'importazione in blocco: {e}", 'error param')
    finally:
        os.remove(zip_path)

    return redirect(url_for('configura_parametri'))

@app.route('/recover_tutti', methods=['POST'])
@login_required
def recover_tutti():
    dir_cwd = os.getcwd()
    old_files = [f for f in os.listdir(dir_cwd) if f.endswith('.old')]
    if not old_files:
        flash("Nessun file .old trovato.", 'info param')
        return redirect(url_for('configura_parametri'))

    for oldf in old_files:
        base_original = oldf[:-4]
        old_path = os.path.join(dir_cwd, oldf)
        original_path = os.path.join(dir_cwd, base_original)
        if os.path.exists(original_path):
            os.remove(original_path)
        os.rename(old_path, original_path)
    flash("Recuperate tutte le versioni .old per i file trovati.", 'success param')
    return redirect(url_for('configura_parametri'))




# inicializa filtros na sessão PARA A GALERIA
def _get_filters():
    return session.get("filters", {"tags": [], "dirs": [], "colors": []})

def _set_filters(f):
    session["filters"] = f

@app.route("/update_filters", methods=["POST"])
def update_filters():
    data = request.get_json(force=True)
    f = _get_filters()

    tipo  = data.get("type")
    valor = data.get("value")
    action = data.get("action")

    if tipo not in f:
        f[tipo] = []

    if action == "add" and valor not in f[tipo]:
        f[tipo].append(valor)
    elif action == "remove" and valor in f[tipo]:
        f[tipo].remove(valor)

    _set_filters(f)

    # agora filtra imagens combinando os filtros acumulados
    imgs = filtra_imagens(f)
    return jsonify({"filters": f, "images": imgs})

@app.route("/clear_filters", methods=["POST"])
def clear_filters():
    _set_filters({"tags": [], "dirs": [], "colors": []})
    imgs = filtra_imagens(_get_filters())
    return jsonify({"filters": _get_filters(), "images": imgs})
# ======================== LOGS ========================
@app.route('/logs')
@login_required
def view_logs():
    from datetime import datetime
    query_date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    search_term = request.args.get('q', '')
    today_str = datetime.now().strftime('%Y-%m-%d')
    if query_date == today_str:
        log_file = os.path.join('logs', 'app.log')
    else:
        log_file = os.path.join('logs', f'app.log.{query_date}')
    logs = []
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if search_term.lower() in line.lower():
                    logs.append(line.strip())
    except Exception as e:
        logs = [f"Errore nel leggere il file di log: {e}"]
    stats = {'total': 0, 'DEBUG': 0, 'INFO': 0, 'WARNING': 0, 'ERROR': 0}
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                stats['total'] += 1
                if 'ERROR' in line:
                    stats['ERROR'] += 1
                elif 'WARNING' in line:
                    stats['WARNING'] += 1
                elif 'INFO' in line:
                    stats['INFO'] += 1
                elif 'DEBUG' in line:
                    stats['DEBUG'] += 1
    except Exception as e:
        stats = {}
    return render_template('logs.html', logs=logs, date=query_date, search_term=search_term, stats=stats)

# ======================== GESTIONE UTENTI ========================
@app.route('/gestione_utenti', methods=['GET', 'POST'])
@login_required
def gestione_utenti():
    if request.method == 'POST':
        nome_nuovo = request.form.get('nome_nuovo', '').strip()
        password_nuova = request.form.get('password_nuova', '').strip()
        if not nome_nuovo or not password_nuova:
            flash('Nome utente e password sono obbligatori.', 'error')
            return redirect(url_for('gestione_utenti'))
        utenti = carica_utenti()
        for u in utenti:
            if u['username'] == nome_nuovo:
                flash('Nome utente già esistente!', 'error')
                return redirect(url_for('gestione_utenti'))
        utenti.append({'username': nome_nuovo, 'password': password_nuova})
        salva_utenti(utenti)
        flash('Utente aggiunto con successo!', 'success')
        return redirect(url_for('gestione_utenti'))
    else:
        utenti = carica_utenti()
        return render_template('gestione_utenti.html', utenti=utenti)

@app.route('/rimuovi_utente', methods=['POST'])
@login_required
def rimuovi_utente():
    username_rimuovere = request.form.get('username_rimuovere')
    if not username_rimuovere:
        flash('Nome utente non fornito.', 'error')
        return redirect(url_for('gestione_utenti'))

    utenti = carica_utenti()
    nuovi = [u for u in utenti if u['username'] != username_rimuovere]
    if len(nuovi) == len(utenti):
        flash('Utente non trovato.', 'error')
    else:
        salva_utenti(nuovi)
        flash('Utente rimosso con successo!', 'success')
    return redirect(url_for('gestione_utenti'))

# ==================== AGENDAMENTO (APSCHEDULER) ====================
from apscheduler.schedulers.background import BackgroundScheduler

reindicizzazione_programmata_progress = {
    'percentuale': 0,
    'log': [],
    'completed': False
}
reindicizzazione_programmata_lock = threading.Lock()

@app.route('/stato_reindicizzazione_programmata')
@login_required
def stato_reindicizzazione_programmata():
    with reindicizzazione_programmata_lock:
        return jsonify(reindicizzazione_programmata_progress.copy())

def job_reindicizza_programmata():
    logger.debug("job_reindicizza_programmata() disparado!")
    with reindicizzazione_programmata_lock:
        reindicizzazione_programmata_progress['percentuale'] = 0
        reindicizzazione_programmata_progress['log'] = ["Reindicizzazione programmata avviata..."]
        reindicizzazione_programmata_progress['completed'] = False

    def run_scheduled():
        try:
            lista_paths = [d["path"] for d in directories_indicizzate]
            indicizza_immagini(lista_paths, file_indice_faiss, file_percorsi, indicizzazione_interrompida)
            with reindicizzazione_programmata_lock:
                reindicizzazione_programmata_progress['percentuale'] = 100
                reindicizzazione_programmata_progress['log'].append("Reindicizzazione programmata completata con successo.")
                reindicizzazione_programmata_progress['completed'] = True
        except Exception as e:
            with reindicizzazione_programmata_lock:
                reindicizzazione_programmata_progress['log'].append(f"Errore: {e}")
                reindicizzazione_programmata_progress['percentuale'] = 100
                reindicizzazione_programmata_progress['completed'] = True
            logger.error("Errore reindicizza programmada: %s", e)

    t = threading.Thread(target=run_scheduled)
    t.start()

def start_scheduler():
    logger.debug("start_scheduler() -> Carregando pianificazione...")
    carica_pianificazione()
    days = schedule_data.get("days", [])
    hour_str = schedule_data.get("hour", "00:00")

    if not days or not hour_str:
        logger.debug("Nenhum dia/hora programado => Scheduler não criou job")
        return

    day_map_ita = {
        "Lunedì": "mon",
        "Martedì": "tue",
        "Mercoledì": "wed",
        "Giovedì": "thu",
        "Venerdì": "fri",
        "Sabato": "sat",
        "Domenica": "sun"
    }
    days_en = []
    for d in days:
        if d in day_map_ita:
            days_en.append(day_map_ita[d])
    day_of_week = ",".join(days_en)

    h, m = 0, 0
    try:
        part = hour_str.split(":")
        h = int(part[0])
        m = int(part[1]) if len(part) > 1 else 0
    except:
        pass

    logger.debug(f"Programando reindex com day_of_week={day_of_week}, hour={h}, minute={m}")
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        func=job_reindicizza_programmata,
        trigger='cron',
        day_of_week=day_of_week,
        hour=h,
        minute=m,
        id='reindex_agendado',
        replace_existing=True
    )
    scheduler.start()
    logger.debug("APScheduler iniciado. Jobs atuais: %s", scheduler.get_jobs())

@app.route('/programmazione_indicizzazione')
@login_required
def programmazione_indicizzazione():
    carica_pianificazione()
    return render_template('programmazione_indicizzazione.html', schedule=schedule_data)

@app.route('/salva_programmazione_indicizzazione', methods=['POST'])
@login_required
def salva_programmazione_indicizzazione():
    global schedule_data
    days_selected = request.form.getlist('days')
    hour_selected = request.form.get('ora_programmata', '00:00')
    new_schedule = {"days": days_selected, "hour": hour_selected}
    with schedule_lock:
        schedule_data = new_schedule
        salva_pianificazione(schedule_data)
    logger.debug("Programma salvata => start_scheduler() novamente")
    start_scheduler()
    flash('Programmazione salvata con successo.', 'success')
    return redirect(url_for('programmazione_indicizzazione'))
# -----------------------------------------------------------------------
# rota /configura_categorie (exemplo)
@ app.route("/configura_categorie", methods=["GET", "POST"])
@login_required
def configura_categorie():
    if request.method == "POST":
        tipo = request.form["tipo_config"]      # 'ricerca' | 'generativa'
        data = json.loads(request.form["categorie_data"])
        path = "static/categorie.json" if tipo=="ricerca" else \
               "static/categorie_generativa.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        flash("Salvato!", "success")
        return redirect(url_for("configura_categorie"))

    # GET – carrega os dois arquivos
    def _load(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    return render_template(
        "configura_categorie.html",
        categorie_search=json.dumps(_load("static/categorie.json")),
        categorie_gen=json.dumps(_load("static/categorie_generativa.json")),
    )


# ============================================
# =  SCANSIONE / CONVERSIONE AUTOMATICA     =
# ============================================

@app.route("/stato_conversor_auto")
@login_required
def stato_conversor_auto():
    # à prova de KeyError mesmo se 'stats' ainda não existir
    with auto_conv_lock:
        logs  = auto_conversion_status.get("log", [])[-40:]
        stats = dict(auto_conversion_status.get("stats", {}))
    return jsonify({"new_logs": logs, "stats": stats})

@app.route("/forza_scansione_conversor", methods=["POST"])
@login_required
def forza_scansione_conversor():
    _auto_log("[MANUALE] Scansione richiesta…")
    threading.Thread(target=_run_one_scan, daemon=True).start()
    return ("", 204)


def _run_one_scan():
    from datetime import datetime

    cfg = load_config()
    regole = cfg.get("conversion_watch", [])
    if not regole:
        _auto_log("[SCAN] Nessuna regola configurata.")
        with auto_conv_lock:
            s = auto_conversion_status.setdefault("stats", {})
            s.setdefault("last_scan", "-")
        return

    WATCH_EXTS = (".psd", ".psb", ".ifd", ".tif", ".tiff",
              ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".jp2")


    # --- carrega todos os caminhos indexados (imagens finais) uma vez ---
    indexed_paths: set[str] = set()
    try:
        if os.path.exists(file_percorsi):
            with open(file_percorsi, "rb") as f:
                indexed_paths = set(pickle.load(f))
    except Exception:
        indexed_paths = set()

    # mapa: diretório de saída -> nomes de arquivo já indexados nele
    indexed_by_outdir: dict[str, set[str]] = {}
    for p in indexed_paths:
        p_norm = os.path.normpath(p)
        outdir = os.path.dirname(p_norm)
        fname  = os.path.basename(p_norm)
        indexed_by_outdir.setdefault(outdir, set()).add(fname)

    # helpers de coerção
    def _to_int_or_none(v):
        if v in (None, "", "None"):
            return None
        try:
            return int(v)
        except Exception:
            return None

    def _to_bool(v):
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in ("1", "true", "on", "yes", "y", "t")

    # existência robusta (ignora arquivos 0 bytes/fantasmas)
    def _file_exists_ok(path: str) -> bool:
        try:
            if not os.path.isfile(path):
                return False
            return os.path.getsize(path) > 1024  # >1KB
        except Exception:
            return False

    local_scanned = local_converted = local_skip_ex = local_skip_idx = local_err = 0

    for rule in regole:
        in_dir  = rule.get("input_dir")
        out_dir = rule.get("output_dir")
        opts    = (rule.get("options") or {})

        if not in_dir or not out_dir:
            continue
        if not os.path.exists(in_dir):
            _auto_log(f"[WARN] Input non esiste: {in_dir}")
            continue

        # garante saída
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            _auto_log(f"[ERRORE] Output non accessibile: {out_dir} ({e})")
            local_err += 1
            continue

        # --- prepara opções do conversor (NÃO passa 'quality' no __init__) ---
        try:
            from conversor import ConversionOptions, _process

            ctor_kwargs = {}
            if "width" in opts:   ctor_kwargs["width"]   = _to_int_or_none(opts.get("width"))
            if "height" in opts:  ctor_kwargs["height"]  = _to_int_or_none(opts.get("height"))
            if "workers" in opts: ctor_kwargs["workers"] = int(opts.get("workers") or 1)

            if "square" in opts:      ctor_kwargs["square"]      = _to_bool(opts.get("square"))
            if "keep_aspect" in opts: ctor_kwargs["keep_aspect"] = _to_bool(opts.get("keep_aspect"))

            ctor_kwargs["skip_duplicates"] = True  # sentinel sempre pula duplicados
            if "icc_profile" in opts:   ctor_kwargs["icc_profile"]   = opts.get("icc_profile") or None
            if "output_format" in opts: ctor_kwargs["output_format"] = str(opts.get("output_format") or "jpg").lower()
            else:                       ctor_kwargs["output_format"] = "jpg"

            conv_opts = ConversionOptions(**ctor_kwargs)

            # pós-construtor opcionais (se a classe tiver os atributos)
            if "colorize_ia" in opts and hasattr(conv_opts, "colorize_ia"):
                setattr(conv_opts, "colorize_ia", _to_bool(opts.get("colorize_ia")))
            if "quality" in opts and hasattr(conv_opts, "quality"):
                q = _to_int_or_none(opts.get("quality"))
                if q is not None:
                    try:
                        setattr(conv_opts, "quality", int(q))
                    except Exception:
                        pass

        except Exception as e:
            _auto_log(f"[ERRORE] Opzioni non valide per la regola {in_dir} → {out_dir}: {e}")
            local_err += 1
            continue

        # set de nomes já indexados **neste** out_dir
        indexed_names_here = indexed_by_outdir.get(os.path.normpath(out_dir), set())

        for root, _, files in os.walk(in_dir):
            for fname in files:
                if not fname.lower().endswith(WATCH_EXTS):
                    continue

                src = os.path.join(root, fname)
                base = os.path.splitext(fname)[0]
                out = os.path.join(out_dir, base + ".jpg")

                local_scanned += 1

                # 1) pular se já existe um arquivo OK na pasta de saída (não conta arquivo fantasma)
                if _file_exists_ok(out):
                    local_skip_ex += 1
                    _auto_log(f"[SKIP esiste] {src} → {out}")
                    continue

                # 2) pular se JÁ ESTÁ INDICIZZATO **neste mesmo out_dir**
                if os.path.basename(out) in indexed_names_here:
                    local_skip_idx += 1
                    _auto_log(f"[SKIP indicizzato] {src} → {out}")
                    continue

                # 3) tenta converter
                try:
                    # agora passamos explicitamente o out_dir
                    status, path_ret, detail = _process(src, conv_opts, dest_dir=out_dir)

                    if status == "converted":
                        local_converted += 1
                        _auto_log(f"[OK] {src} → {out}")
                    elif status == "skipped":
                        local_skip_ex += 1
                        _auto_log(f"[SKIP esiste] {src} → {out}")
                    else:
                        local_err += 1
                        _auto_log(f"[ERRORE] {src}: {detail or 'errore sconosciuto'}")
                except Exception as e:
                    local_err += 1
                    _auto_log(f"[ERRORE] {src}: {e}")

    # acumula counters globais
    with auto_conv_lock:
        s = auto_conversion_status.setdefault("stats", {})
        s["scanned"]          = s.get("scanned", 0) + local_scanned
        s["converted"]        = s.get("converted", 0) + local_converted
        s["skipped_existing"] = s.get("skipped_existing", 0) + local_skip_ex
        s["skipped_indexed"]  = s.get("skipped_indexed", 0) + local_skip_idx
        s["errors"]           = s.get("errors", 0) + local_err
        s["last_scan"]        = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def start_conversion_sentinel():
    """
    Loop sentinella che richiama _run_one_scan() ogni 10s.
    Mantiene 'auto_conversion_status.running' e inizializza 'stats'.
    """
    def loop():
        _auto_log("[SENTINELLA] Avviato. Se 'pyvips' non è disponibile, uso Pillow (OK).")
        while True:
            try:
                with auto_conv_lock:
                    # garante estrutura 'stats'
                    st = auto_conversion_status.setdefault("stats", {})
                    st.setdefault("scanned", 0)
                    st.setdefault("converted", 0)
                    st.setdefault("skipped_existing", 0)
                    st.setdefault("skipped_indexed", 0)
                    st.setdefault("errors", 0)
                    st.setdefault("last_scan", "-")
                    auto_conversion_status["running"] = True

                _run_one_scan()

            except Exception as e:
                _auto_log(f"[SENTINELLA] Errore di ciclo: {e}")

            finally:
                with auto_conv_lock:
                    auto_conversion_status["running"] = False

            time.sleep(int(os.getenv("CONV_SCAN_INTERVAL", "60")))  # default: 60s


    t = threading.Thread(target=loop, daemon=True)
    t.start()

if __name__ == '__main__':
    start_conversion_sentinel()
    ensure_default_user()
    logger.debug("[DEBUG] Chamando start_scheduler() p/ APScheduler.")
    start_scheduler()
    from waitress import serve
    logger.debug("[DEBUG] Subindo waitress server na porta 5000...")
    serve(app, host='0.0.0.0', port=5000, threads=8)
