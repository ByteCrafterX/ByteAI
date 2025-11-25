#!/usr/bin/env python3
# =========================================================
#  conversor.py  –  Super-Converter (JPEG, crop-cover + IA)
#  Modo seguro (NAS/contêiner) com SAFE MODE turbinado:
#  - PSD/PSB/TIFF grandes => ImageMagick com limites (memory/map/disk)
#  - SAFE MODE multi-pass (25% → 12.5% → 6.25%) p/ arquivos gigantes
#  - Fallback Pillow thumbnail (4096px) se tudo falhar
#  - Pillow abre arquivos reduzindo logo no começo para poupar RAM
#  - Paralelismo conservador (1 por padrão), ajustável por env
# =========================================================
from __future__ import annotations

import os
import logging
import traceback
import importlib.util
import warnings
import tempfile
import uuid
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Optional, List

import numpy as np

# ---------------- log base ----------------
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    level=logging.INFO)
log = logging.getLogger("conversor")

# ======= Parâmetros/limites rigorosos via env =======
SAFE_MAX_LONG   = int(os.getenv("CONV_SAFE_LONG_SIDE", "2400"))   # lado maior ao abrir
HUGE_MAX_PIXELS = int(os.getenv("CONV_HUGE_PIXELS", "60000000"))  # 60 MP
HUGE_MAX_BYTES  = int(os.getenv("CONV_HUGE_BYTES",  "150000000")) # 150 MB (tamanho de arquivo)
IM_BIN          = os.getenv("CONV_IM_BIN", "magick")              # "magick" (Windows/IM7) ou "convert" (IM6)

IM_LIMIT_MEM  = os.getenv("CONV_IM_LIMIT_MEM",  "512MiB")
IM_LIMIT_MAP  = os.getenv("CONV_IM_LIMIT_MAP",  "1GiB")
IM_LIMIT_DISK = os.getenv("CONV_IM_LIMIT_DISK", "2GiB")

SAFE_TMPDIR = os.getenv("CONV_SAFE_TMPDIR", tempfile.gettempdir())

# ---- Desabilite pyvips por padrão (só usa se estiver 100% ok) ----
USE_VIPS = False
pyvips = None
try:
    spec = importlib.util.find_spec("pyvips")
    if spec:
        import pyvips as _pv
        _pv.version(0)
        pyvips = _pv
        # Se quiser permitir pyvips, troque para True. Mantemos False por segurança.
        USE_VIPS = False
        log.info("pyvips encontrado, mas desativado por padrão (fallback = Pillow/IM).")
except Exception as e:
    warnings.filterwarnings("ignore", message="pyvips off*")
    import logging as _lg
    _lg.getLogger("pyvips").setLevel(_lg.ERROR)
    log.warning("pyvips indisponível (%s) → Pillow/IM", e)

# ---------------- Pillow ------------------
from PIL import Image, ImageCms, ImageFile, ImageOps
import imageio.v2 as imageio
try:
    from psd_tools import PSDImage
except ImportError:
    PSDImage = None

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============ Helpers de abertura segura ============
def _pick_first_frame_if_needed(img: Image.Image) -> Image.Image:
    try:
        if getattr(img, "n_frames", 1) > 1:
            img.seek(0)
    except Exception:
        pass
    return img


def _integer_reduce_factor(w: int, h: int, target: int) -> int:
    # fator inteiro 1,2,4,8... para reduzir antes de materializar
    if target <= 0:
        return 1
    long_side = max(w, h)
    f = 1
    while (long_side // (f * 2)) >= target:
        f *= 2
    return max(1, f)


def open_image_safely(path: str, target_long_side: int = SAFE_MAX_LONG) -> Image.Image:
    """
    Abre reduzindo o decode quando possível (draft/reduce/thumbnail),
    para gastar o mínimo de RAM antes de cropar/redimensionar.
    """
    img = Image.open(path)
    img = _pick_first_frame_if_needed(img)

    try:
        w, h = img.size
        factor = _integer_reduce_factor(w, h, target_long_side)

        # pedir decode menor (JPEG/WebP)
        try:
            img.draft("RGB", (target_long_side, target_long_side))
        except Exception:
            pass

        # TIFF piramidal?
        if img.format == "TIFF" and hasattr(img, "reduce") and factor > 1:
            try:
                img = img.reduce(factor)
            except Exception:
                pass

        if max(img.size) > target_long_side:
            img.thumbnail((target_long_side, target_long_side), Image.LANCZOS)

        img = ImageOps.exif_transpose(img)
        img.load()
        return img

    except MemoryError:
        # fallback agressivo
        try:
            img.close()
        except Exception:
            pass
        img = Image.open(path)
        img = _pick_first_frame_if_needed(img)
        tiny = max(512, min(1024, target_long_side // 2))
        try:
            img.draft("RGB", (tiny, tiny))
        except Exception:
            pass
        if hasattr(img, "reduce"):
            try:
                img = img.reduce(_integer_reduce_factor(*img.size, tiny))
            except Exception:
                pass
        if max(img.size) > tiny:
            img.thumbnail((tiny, tiny), Image.LANCZOS)
        img = ImageOps.exif_transpose(img)
        img.load()
        return img


# ============ Detecta “gigantesco” ============
def _is_huge_image(path: str) -> bool:
    try:
        st = os.stat(path)
        if st.st_size >= HUGE_MAX_BYTES:
            return True
    except Exception:
        pass
    try:
        with Image.open(path) as im:
            w, h = im.size
            if (w * h) >= HUGE_MAX_PIXELS:
                return True
    except Exception:
        # não conseguiu abrir p/ medir: não marca por pixels
        pass
    return False


# =========================================================
#  SAFE MODE CORE — MULTI PASS RESIZE (25% → 12.5% → 6.25%)
# =========================================================
def _safe_reduce_im(inp: str, out_tmp: str, percent: float) -> bool:
    """
    Reduz o PSD/TIFF via ImageMagick sem montar tudo em RAM.
    Retorna True se deu certo, False se falhou.
    """
    bin_ = shutil.which(IM_BIN) or shutil.which("convert")
    if not bin_:
        log.error("ImageMagick não encontrado para SAFE MODE!")
        return False

    cmd = [
        bin_,
        "-limit", "memory", IM_LIMIT_MEM,
        "-limit", "map",    IM_LIMIT_MAP,
        "-limit", "disk",   IM_LIMIT_DISK,
        "-quiet",
        f"{inp}[0]",
        "-resize", f"{percent}%",
        "-colorspace", "sRGB",
        "-flatten",
        f"tiff:{out_tmp}",
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        log.warning(f"SAFE REDUCE {percent}% falhou: {proc.stderr.strip()}")
        return False

    return os.path.exists(out_tmp)


def safe_convert_huge(inp: str, out: str, opts) -> None:
    """
    SAFE MODE:
      Tenta converter arquivos gigantes com multi-pass:
        25% → 12.5% → 6.25%
    Garante que SEMPRE produz JPG sem explodir RAM.
    """
    base = os.path.basename(inp)
    safe_tmp = os.path.join(SAFE_TMPDIR, f"safe_{uuid.uuid4().hex}.tif")

    for pct in (25, 12.5, 6.25):
        try:
            if _safe_reduce_im(inp, safe_tmp, pct):
                log.info(f"[SAFE] {base} → reduzido para {pct}% com sucesso")
                try:
                    _convert_with_imagemagick(safe_tmp, out, opts)
                    try:
                        os.remove(safe_tmp)
                    except Exception:
                        pass
                    return
                except Exception as e:
                    log.warning(f"[SAFE] conversão IM falhou após reduzir {pct}%: {e}")
        except Exception as e:
            log.warning(f"[SAFE] erro inesperado {pct}%: {e}")

    # fallback final — baixo consumo
    log.warning(f"[SAFE-FALLBACK] Usando Pillow thumbnail para {base}")
    try:
        pil = Image.open(inp)
        pil.thumbnail((4096, 4096), Image.LANCZOS)
        pil = pil.convert("RGB")
        pil.save(out, quality=92)
    except Exception as e:
        raise RuntimeError(f"Fallback total falhou: {e}")


# ============ ImageMagick com limites ============
def _im_bin() -> str:
    # usa "magick" se existir; senão tenta "convert"
    if shutil.which(IM_BIN):
        return IM_BIN
    if shutil.which("convert"):
        return "convert"
    # se não tem IM, falha (melhor saber do que estourar RAM)
    raise RuntimeError("ImageMagick não encontrado no contêiner (instale imagemagick).")


def _build_im_resize_args(opts) -> list:
    # cover/crop igual ao Pillow
    W = opts.width if getattr(opts, "width", None) not in (None, "", "None") else 1200
    H = opts.height if getattr(opts, "height", None) not in (None, "", "None") else 1200
    try:
        W = int(W)
    except Exception:
        W = 1200
    try:
        H = int(H)
    except Exception:
        H = 1200

    q = getattr(opts, "quality", 92)
    try:
        q = int(q)
    except Exception:
        q = 92

    if getattr(opts, "square", False):
        S = min(W, H)
        return [
            "-resize", f"{S}x{S}^",
            "-gravity", "center",
            "-extent", f"{S}x{S}",
            "-strip",
            "-quality", str(q),
        ]
    else:
        if W and H:
            return [
                "-resize", f"{W}x{H}^",
                "-gravity", "center",
                "-extent", f"{W}x{H}",
                "-strip",
                "-quality", str(q),
            ]
        else:
            side = W or H or 1200
            return [
                "-resize", f"{side}x{side}",
                "-strip",
                "-quality", str(q),
            ]


def _convert_with_imagemagick(inp: str, out: str, opts) -> None:
    """
    Converte com IM: pega a 1ª página/camada ([0]), flattens, aplica cover/crop e salva JPEG.
    Usa limites p/ não estourar: memory/map/disk.
    Se falhar por memória/cache → chama SAFE MODE.
    """
    bin_ = _im_bin()
    limits = [
        "-limit", "memory", IM_LIMIT_MEM,
        "-limit", "map",    IM_LIMIT_MAP,
        "-limit", "disk",   IM_LIMIT_DISK,
        "-quiet",
    ]
    resize_args = _build_im_resize_args(opts)

    # escreve para um tmp primeiro para evitar arquivos quebrados em falha
    tmp_out = out + ".imtmp.jpg"
    if os.path.exists(tmp_out):
        try:
            os.remove(tmp_out)
        except Exception:
            pass

    cmd = [
        bin_,
        *limits,
        f"{inp}[0]",
        "-colorspace", "sRGB",
        "-flatten",
        *resize_args,
        f"jpg:{tmp_out}",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if proc.returncode != 0 or not os.path.exists(tmp_out):
        err = (proc.stderr or proc.stdout or "").strip().lower()

        # Se falhou por causa de memória/cache —> ativa SAFE MODE
        if "cache" in err or "memory" in err or "exhausted" in err:
            log.warning(f"[SAFE-TRIGGER] IM falhou por memória em {inp}")
            safe_convert_huge(inp, out, opts)
            return

        raise RuntimeError(f"ImageMagick falhou ({proc.returncode}): {err}")

    # move atômico
    os.replace(tmp_out, out)


# -------------- dataclass -----------------
@dataclass(slots=True)
class ConversionOptions:
    width: Optional[int]  = None
    height: Optional[int] = None
    square: bool          = False
    keep_aspect: bool     = True
    skip_duplicates: bool = True
    icc_profile: Optional[str] = None
    workers: int          = int(os.getenv("CONV_WORKERS_DEFAULT", "1"))
    output_format: str    = "jpg"
    colorize_ia: bool     = False    # ← IA-colorização
    quality: int          = 92


# -------- plugin architecture -------------
class FormatHandler:
    SUPPORTED: set[str] = set()

    def can_handle(self, ext: str) -> bool:
        return ext in self.SUPPORTED

    def convert(self, inp: str, out: str, opts: ConversionOptions):
        raise NotImplementedError


_handlers: List[FormatHandler] = []


def register(cls):
    _handlers.append(cls())
    return cls


def find_handler(ext: str) -> FormatHandler:
    for h in _handlers:
        if h.can_handle(ext):
            return h
    return _handlers[-1]  # fallback Pillow sempre presente


def _center_crop(pil: Image.Image, w: int, h: int) -> Image.Image:
    left = (pil.width - w) // 2
    top = (pil.height - h) // 2
    return pil.crop((left, top, left + w, top + h))


# ---------------- IA Colorize -------------
def _apply_colorize(pil: Image.Image) -> Image.Image:
    if pil.mode != "L":
        return pil.convert("RGB") if pil.mode != "RGB" else pil
    try:
        from deoldify.visualize import get_image_colorizer

        colorizer = get_image_colorizer(artistic=True)
        tmpin = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.png")
        pil.save(tmpin)
        colorizer.colorize_from_file(tmpin, render_factor=35, watermarked=False)
        tmpout = tmpin.replace(".png", "_color.png")
        pil_col = Image.open(tmpout)
        return pil_col.convert("RGB")
    except Exception as e:
        log.warning("DeOldify indisponível (%s) → gradiente", e)
        arr = np.asarray(pil, dtype=np.uint8)
        col = np.stack(
            [arr, (arr * 0.9).astype(np.uint8), (arr * 0.7).astype(np.uint8)], axis=-1
        )
        return Image.fromarray(col, "RGB")


# ---------- VIPS handler (desativado por padrão) ----------
@register
class VipsHandler(FormatHandler):
    SUPPORTED = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".tif",
        ".tiff",
        ".ifd",
        ".avif",
        ".heic",
    }

    def can_handle(self, ext):
        return USE_VIPS and ext in self.SUPPORTED

    def convert(self, inp, out, opts):
        # Mesmo com VIPS, se for gigante, força SAFE MODE
        if _is_huge_image(inp):
            safe_convert_huge(inp, out, opts)
            return
        img = pyvips.Image.new_from_file(inp, access="sequential")
        if opts.icc_profile:
            img = img.icc_transform(opts.icc_profile, "srgb")
        img = _resize_vips_cover(img, opts)
        img.write_to_file(out)


def _resize_vips_cover(img, opts):
    W, H = opts.width, opts.height
    if W and H:
        scale = max(W / img.width, H / img.height)
        img = img.resize(scale, scale)
        x = (img.width - W) // 2
        y = (img.height - H) // 2
        img = img.crop(x, y, W, H)
    elif W or H:
        img = img.thumbnail_image(W or H)
    return img


# ---------- PSD / PSB ----------
@register
class PSDHandler(FormatHandler):
    SUPPORTED = {".psd", ".psb"}

    def can_handle(self, ext):
        return PSDImage and ext in self.SUPPORTED

    def convert(self, inp, out, opts):
        base = os.path.basename(inp)

        # 1) Se for gigante OU psd_tools indisponível -> SAFE MODE direto
        if _is_huge_image(inp) or not PSDImage:
            log.info(f"[SAFE] {base}: PSD enorme ou psd_tools ausente — usando SAFE MODE")
            safe_convert_huge(inp, out, opts)
            return

        # 2) Tenta compor com psd_tools. Se falhar/memory -> SAFE MODE
        try:
            psd = PSDImage.open(inp)
            pil = psd.topil() if hasattr(psd, "topil") else psd.compose()
        except MemoryError:
            log.warning(f"[SAFE] {base}: MemoryError ao abrir psd_tools — SAFE MODE")
            safe_convert_huge(inp, out, opts)
            return
        except Exception as e:
            log.warning(f"[SAFE] {base}: psd_tools falhou ({e}) — SAFE MODE")
            safe_convert_huge(inp, out, opts)
            return

        # Redução preventiva antes de qualquer outra coisa
        target = min(
            max(getattr(opts, "width", 0) or 0, getattr(opts, "height", 0) or 0)
            or SAFE_MAX_LONG,
            SAFE_MAX_LONG,
        )
        if max(pil.size) > target:
            pil.thumbnail((target, target), Image.LANCZOS)
        pil = pil.convert("RGB")
        _pil_cover(pil, out, opts)


# ---------- ImageIO ----------
@register
class ImageIOHandler(FormatHandler):
    SUPPORTED = {".gif", ".bmp", ".jp2", ".ppm"}

    def convert(self, inp, out, opts):
        target = min(
            max(getattr(opts, "width", 0) or 0, getattr(opts, "height", 0) or 0)
            or SAFE_MAX_LONG,
            SAFE_MAX_LONG,
        )
        pil = open_image_safely(inp, target_long_side=target)
        if pil.mode not in ("RGB", "RGBA", "LA", "L"):
            pil = pil.convert("RGBA")
        _pil_cover(pil, out, opts)


# ---------- Pillow fallback ----------
@register
class PILHandler(FormatHandler):
    SUPPORTED = {".tif", ".tiff"}  # TIFF cai aqui também

    def can_handle(self, ext):
        return True  # fallback geral

    def convert(self, inp, out, opts):
        ext = os.path.splitext(inp)[1].lower()
        base = os.path.basename(inp)

        # TIFF enorme vai direto para SAFE MODE
        if ext in (".tif", ".tiff") and _is_huge_image(inp):
            log.info(f"[SAFE] TIFF gigante — usando SAFE MODE: {inp}")
            safe_convert_huge(inp, out, opts)
            return

        target = min(
            max(getattr(opts, "width", 0) or 0, getattr(opts, "height", 0) or 0)
            or SAFE_MAX_LONG,
            SAFE_MAX_LONG,
        )
        try:
            pil = open_image_safely(inp, target_long_side=target)
        except MemoryError:
            log.warning(f"[SAFE] {base}: MemoryError ao abrir TIFF — SAFE MODE")
            safe_convert_huge(inp, out, opts)
            return

        if pil.mode not in ("RGB", "RGBA", "LA", "L"):
            pil = pil.convert("RGBA")
        _pil_cover(pil, out, opts)


# --------- cover/crop comum ----------
def _pil_cover(pil: Image.Image, out: str, opts: ConversionOptions):
    # IA colorize
    if getattr(opts, "colorize_ia", False):
        pil = _apply_colorize(pil)

    # ICC (best-effort)
    if getattr(opts, "icc_profile", None):
        try:
            pil = ImageCms.profileToProfile(
                pil, opts.icc_profile, opts.icc_profile
            )
        except Exception:
            pass

    # Dimensões alvo (defaults)
    W, H = opts.width, opts.height
    try:
        W = int(W) if W not in (None, "", "None") else 1200
    except Exception:
        W = 1200
    try:
        H = int(H) if H not in (None, "", "None") else 1200
    except Exception:
        H = 1200

    if opts.square:
        if W and not H:
            H = W
        elif H and not W:
            W = H
        if W and H:
            S = min(W, H)
            scale = max(S / pil.width, S / pil.height)
            pil = pil.resize(
                (
                    max(1, int(round(pil.width * scale))),
                    max(1, int(round(pil.height * scale))),
                ),
                Image.LANCZOS,
            )
            pil = _center_crop(pil, S, S)
        else:
            side = min(pil.width, pil.height)
            pil = _center_crop(pil, side, side)
    else:
        if W and H:
            if opts.keep_aspect:
                scale = max(W / pil.width, H / pil.height)
                pil = pil.resize(
                    (
                        max(1, int(round(pil.width * scale))),
                        max(1, int(round(pil.height * scale))),
                    ),
                    Image.LANCZOS,
                )
                pil = _center_crop(pil, W, H)
            else:
                pil = pil.resize((W, H), Image.LANCZOS)
        elif W or H:
            tw = W or pil.width
            th = H or pil.height
            if opts.keep_aspect:
                pil.thumbnail((tw, th), Image.LANCZOS)
            else:
                pil = pil.resize((tw, th), Image.LANCZOS)

    if out.lower().endswith((".jpg", ".jpeg")) and pil.mode != "RGB":
        pil = pil.convert("RGB")

    quality = getattr(opts, "quality", 92)
    try:
        quality = int(quality)
    except Exception:
        quality = 92

    pil.save(out, quality=quality)


# ---------- worker ----------
def _process(path, opts, dest_dir=None):
    ext = os.path.splitext(path)[1].lower()
    base = os.path.basename(path)
    base_noext = os.path.splitext(base)[0]

    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)
        out = os.path.join(dest_dir, f"{base_noext}.{opts.output_format}")
    else:
        out = f"{os.path.splitext(path)[0]}.{opts.output_format}"

    # Skip duplicates
    if opts.skip_duplicates and os.path.exists(out):
        return ("skipped", path, None)

    try:
        find_handler(ext).convert(path, out, opts)
        return ("converted", out, None)

    except RuntimeError as e:
        # Erro do ImageMagick → tentar fallback SAFE MODE
        if "cache resources exhausted" in str(e).lower():
            try:
                safe_convert_huge(path, out, opts)
                return ("converted", out, None)
            except Exception as ee:
                return ("error", path, f"SAFE fallback falhou:\n{ee}")

        return ("error", path, traceback.format_exc(limit=6))

    except Exception:
        return ("error", path, traceback.format_exc(limit=6))


# ---------- bulk ----------
def bulk_convert(
    input_path: str,
    opts: ConversionOptions,
    cb: Callable[[int], None] | None = None,
):
    VALID_EXTS = (
        ".psd",
        ".psb",
        ".tif",
        ".tiff",
        ".ifd",
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".gif",
        ".bmp",
        ".jp2",
        ".ppm",
        ".heic",
        ".avif",
    )

    if os.path.isdir(input_path):
        files = [
            os.path.join(r, f)
            for r, _, fs in os.walk(input_path)
            for f in fs
            if os.path.splitext(f)[1].lower() in VALID_EXTS
        ]
    else:
        files = (
            [input_path]
            if os.path.splitext(input_path)[1].lower() in VALID_EXTS
            else []
        )

    total = len(files) or 1
    logs = [f"Avvio conversione: {len(files)} file trovati…"]
    stats = {"converted": 0, "skipped": 0, "error": 0}

    # Paralelismo conservador
    max_w = max(
        1,
        min(int(getattr(opts, "workers", 1) or 1), int(os.getenv("CONV_WORKERS_MAX", "1"))),
    )
    with ProcessPoolExecutor(max_workers=max_w) as ex:
        futs = {ex.submit(_process, f, opts): f for f in files}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                st, path, det = fut.result()
            except Exception as e:
                st, path, det = ("error", "UNKNOWN", str(e))

            stats[st] += 1
            if st == "converted":
                logs.append(f"✅ {path}")
            elif st == "skipped":
                logs.append(f"⏭️ {path}")
            else:
                logs.append(f"❌ {path}\n{det}")
            if cb:
                cb(int(i / total * 100))

    logs.append(
        f"=== Riepilogo ===\nTotale {len(files)} | "
        f"OK {stats['converted']} | Saltati {stats['skipped']} | Errori {stats['error']}"
    )
    return logs, stats


# ---------- CLI ----------
if __name__ == "__main__":
    import click

    @click.command()
    @click.option("-i", "--input", type=click.Path(exists=True), required=True)
    @click.option("-w", "--width", type=int)
    @click.option("-hgt", "--height", type=int)
    @click.option("--workers", type=int, default=int(os.getenv("CONV_WORKERS_DEFAULT", "1")))
    @click.option("--colorize/--no-colorize", default=False, help="IA Colorize")
    def cli(input, width, height, workers, colorize):
        opts = ConversionOptions(
            width=width,
            height=height,
            workers=workers,
            colorize_ia=colorize,
        )
        def prog(p):
            print(f"\r{p}% ", end="", flush=True)
        logs, _ = bulk_convert(input, opts, prog)
        print("\n".join(logs))

    cli()
