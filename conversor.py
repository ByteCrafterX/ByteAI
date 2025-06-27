#!/usr/bin/env python3
# =========================================================
#  conversor.py  –  Super-Converter (JPEG, crop-cover)
# =========================================================
from __future__ import annotations
import os, hashlib, logging, traceback, importlib, importlib.util
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict, Optional

# ---------------- log base ----------------
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    level=logging.INFO)
log = logging.getLogger(__name__)

# -------- pyvips con fallback -------------
USE_VIPS = False
pyvips = None
spec = importlib.util.find_spec("pyvips")
if spec:
    try:
        pyvips = importlib.import_module("pyvips")
        pyvips.version(0)
        USE_VIPS = True
        log.info("pyvips attivo.")
    except Exception as e:
        log.warning("pyvips off (%s) → Pillow", e)
import logging as _lg; _lg.getLogger("pyvips").setLevel(_lg.ERROR)

# ---------------- Pillow ------------------
from PIL import Image, ImageCms, ImageFile
import imageio.v2 as imageio
try:
    from psd_tools import PSDImage
except ImportError:
    PSDImage = None
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# -------------- dataclass -----------------
@dataclass(slots=True)
class ConversionOptions:
    width: Optional[int] = None
    height: Optional[int] = None
    square: bool = False
    keep_aspect: bool = True
    skip_duplicates: bool = True
    icc_profile: Optional[str] = None
    workers: int = os.cpu_count() or 1
    output_format: str = "jpg"          # default JPEG

# -------- plugin architecture ------------
class FormatHandler:
    SUPPORTED:set[str]=set()
    def can_handle(self,ext):return ext in self.SUPPORTED
    def convert(self,inp,out,opts):...

_handlers=[]
def register(cls): _handlers.append(cls()); return cls
def find_handler(ext):
    for h in _handlers:
        if h.can_handle(ext): return h
    return _handlers[-1]

def _center_crop(pil:Image.Image,w:int,h:int)->Image.Image:
    left=(pil.width-w)//2; top=(pil.height-h)//2
    return pil.crop((left, top, left+w, top+h))

# ---------- Vips handler ----------
@register
class VipsHandler(FormatHandler):
    SUPPORTED={".jpg",".jpeg",".png",".webp",".tif",".tiff",".ifd",".avif",".heic"}
    def can_handle(self,ext):return USE_VIPS and ext in self.SUPPORTED
    def convert(self,inp,out,opts):
        img=pyvips.Image.new_from_file(inp,access="sequential")
        if opts.icc_profile: img=img.icc_transform(opts.icc_profile,"srgb")
        img=_resize_vips_cover(img,opts)
        img.write_to_file(out)

def _resize_vips_cover(img,opts):
    W,H=opts.width,opts.height
    if W and H:
        scale=max(W/img.width,H/img.height)
        img=img.resize(scale,scale)
        x=(img.width-W)//2; y=(img.height-H)//2
        img=img.crop(x,y,W,H)
    elif W or H:
        img=img.thumbnail_image(W or H)
    return img

# ---------- PSD/PSB ----------
@register
class PSDHandler(FormatHandler):
    SUPPORTED={".psd",".psb"}
    def can_handle(self,ext):return PSDImage and ext in self.SUPPORTED
    def convert(self,inp,out,opts):
        pil=PSDImage.open(inp).compose().convert("RGBA")
        _pil_cover(pil,out,opts)

# ---------- ImageIO ----------
@register
class ImageIOHandler(FormatHandler):
    SUPPORTED={".gif",".bmp",".jp2",".ppm"}
    def convert(self,inp,out,opts):
        pil=Image.fromarray(imageio.imread(inp))
        _pil_cover(pil,out,opts)

# ---------- Pillow fallback ----------
@register
class PILHandler(FormatHandler):
    SUPPORTED=set()
    def can_handle(self,ext):return True
    def convert(self,inp,out,opts):
        pil=Image.open(inp)
        if pil.mode not in ("RGB","RGBA","LA"): pil=pil.convert("RGBA")
        _pil_cover(pil,out,opts)

def _pil_cover(pil:Image.Image,out:str,opts:ConversionOptions):
    W,H=opts.width,opts.height
    if opts.icc_profile:
        try: pil=ImageCms.profileToProfile(pil,opts.icc_profile,opts.icc_profile)
        except Exception: pass
    if W and H:
        scale=max(W/pil.width,H/pil.height)
        pil=pil.resize((int(pil.width*scale),int(pil.height*scale)),Image.LANCZOS)
        pil=_center_crop(pil,W,H)
    elif W or H:
        pil.thumbnail((W or pil.width,H or pil.height),Image.LANCZOS)
    if out.lower().endswith((".jpg",".jpeg")) and pil.mode!="RGB":
        pil=pil.convert("RGB")
    pil.save(out,quality=92)

# ---------- worker ----------
def _process(path,opts):
    ext=os.path.splitext(path)[1].lower()
    out=f"{os.path.splitext(path)[0]}.{opts.output_format}"
    if opts.skip_duplicates and os.path.exists(out):return("skipped",path,None)
    try:
        find_handler(ext).convert(path,out,opts)
        return("converted",path,None)
    except Exception as e:
        return("error",path,traceback.format_exc(limit=3))

def bulk_convert(input_path,opts,cb=None):
    files=[os.path.join(r,f) for r,_,fs in os.walk(input_path) for f in fs] if os.path.isdir(input_path) else [input_path]
    total=len(files) or 1
    logs=[f"Avvio conversione: {total} file…"]; stats={"converted":0,"skipped":0,"error":0}
    with ProcessPoolExecutor(max_workers=opts.workers) as ex:
        futs={ex.submit(_process,f,opts):f for f in files}
        for i,fut in enumerate(as_completed(futs),1):
            st,path,det=fut.result(); stats[st]+=1
            if st=="converted": logs.append(f"✅ {path}")
            elif st=="skipped": logs.append(f"⏭️ {path}")
            else: logs.append(f"❌ {path}\n{det}")
            if cb: cb(int(i/total*100))
    logs.append(f"=== Riepilogo ===\nTotale {total} | OK {stats['converted']} | Saltati {stats['skipped']} | Errori {stats['error']}")
    return logs,stats

# ---------- CLI ----------
if __name__=="__main__":
    import click
    @click.command()
    @click.option("-i","--input",type=click.Path(exists=True),required=True)
    @click.option("-w","--width",type=int)
    @click.option("-hgt","--height",type=int)
    @click.option("--workers",type=int,default=os.cpu_count())
    def cli(input,width,height,workers):
        opts=ConversionOptions(width=width,height=height,workers=workers)
        def prog(p): print(f"\r{p}% ",end="",flush=True)
        print("\n".join(bulk_convert(input,opts,prog)[0]))
    cli()
