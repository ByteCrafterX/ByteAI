# modello.py – CLIP + OCR-first + busca por embedding + duplicatas
import os
import json
import pickle
import math
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import faiss
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ============================================================
#   CONFIG / SINGLETONS
# ============================================================
_DEVICE = torch.device("cpu")
_MODEL: Optional[CLIPModel] = None
_PROC: Optional[CLIPProcessor] = None

def _get_clip():
    global _MODEL, _PROC
    if _MODEL is None or _PROC is None:
        _MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _PROC  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _MODEL = _MODEL.to(_DEVICE)
        # evita dtype half em CPU
        if _MODEL.dtype == torch.float16:
            _MODEL = _MODEL.to(torch.float32)
    return _MODEL, _PROC


# ============================================================
#   UTILIDADES
# ============================================================
def _norm_text(s: str) -> str:
    import unicodedata, re
    s = unicodedata.normalize("NFKC", s or "")
    s = s.casefold()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _load_faiss(file_indice_faiss: str):
    if not os.path.isfile(file_indice_faiss):
        return None
    return faiss.read_index(file_indice_faiss)

def _load_paths(file_percorsi: str) -> List[str]:
    if not os.path.isfile(file_percorsi):
        return []
    with open(file_percorsi, "rb") as f:
        return pickle.load(f)

def _filter_by_dirs(paths: List[str], dirs: List[str]) -> List[bool]:
    """Retorna máscara True/False indicando se path começa com algum dir selecionado."""
    if not dirs:
        return [True] * len(paths)
    dn = [os.path.normpath(d) for d in dirs]
    out = []
    for p in paths:
        pn = os.path.normpath(p)
        out.append(any(pn.startswith(d) for d in dn))
    return out


# ============================================================
#   FEATURES
# ============================================================
def extrai_features_imagem(img_path: str) -> np.ndarray:
    """Extrai vetor CLIP de uma imagem (normalizado L2)."""
    model, proc = _get_clip()
    with Image.open(img_path).convert("RGB") as im:
        inputs = proc(images=[im], return_tensors="pt").to(_DEVICE)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    arr = feats.cpu().numpy().astype("float32")
    faiss.normalize_L2(arr)
    return arr[0]  # (d,)

def _emb_texto(descr: str) -> np.ndarray:
    model, proc = _get_clip()
    inputs = proc(text=[descr or ""], return_tensors="pt", padding=True).to(_DEVICE)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
    arr = feats.cpu().numpy().astype("float32")
    faiss.normalize_L2(arr)
    return arr[0]


# ============================================================
#   BUSCA POR EMBEDDING (imagem->imagens)
# ============================================================
def cerca_per_embedding(
    emb: np.ndarray,
    file_indice_faiss: str,
    file_percorsi: str,
    directories_selezionate: Optional[List[str]] = None,
    offset: int = 0,
    limit: int = 100,
) -> Tuple[List[Dict[str, Any]], int]:
    """Retorna itens mais similares ao embedding fornecido (cosine/IP)."""
    idx = _load_faiss(file_indice_faiss)
    paths = _load_paths(file_percorsi)
    if idx is None or not paths:
        return [], 0

    if emb.ndim == 1:
        q = emb[None, :]
    else:
        q = emb

    # top-K “largo” para permitir filtro por diretórios
    K = min(len(paths), max(limit + offset, 1000))
    D, I = idx.search(q.astype("float32"), K)
    ranks = I[0].tolist()
    scores = D[0].tolist()

    mask = _filter_by_dirs(paths, directories_selezionate or [])
    filt = [(i, s) for i, s in zip(ranks, scores) if i >= 0 and mask[i]]
    total = len(filt)
    slice_ = filt[offset: offset + limit]

    out = []
    for i, s in slice_:
        p = paths[i]
        out.append({
            "percorso": p,
            "percorso_url": p,
            "score": float(s),
            "extra_ocr_match": False
        })
    return out, total


# ============================================================
#   BUSCA CLIP TEXTO→IMAGEM (fluxo padrão)
# ============================================================
def _cerca_clip_text(
    descrizione: str,
    file_indice_faiss: str,
    file_percorsi: str,
    directories_selezionate: Optional[List[str]],
    offset: int,
    limit: int,
) -> Tuple[List[Dict[str, Any]], int]:
    idx = _load_faiss(file_indice_faiss)
    paths = _load_paths(file_percorsi)
    if idx is None or not paths or not (descrizione or "").strip():
        return [], 0

    emb_t = _emb_texto((descrizione or "").strip())
    return cerca_per_embedding(
        emb_t, file_indice_faiss, file_percorsi,
        directories_selezionate=directories_selezionate,
        offset=offset, limit=limit
    )


# ============================================================
#   OCR-FIRST (helpers com prefixo exclusivo)
# ============================================================
__OCR_PATH = "ocr_metadata.json"

def __ocr_load() -> Dict[str, str]:
    if not os.path.exists(__OCR_PATH):
        return {}
    try:
        with open(__OCR_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def __ocr_score(query_norm: str, text_norm: str) -> float:
    if not query_norm or not text_norm:
        return 0.0
    q = set(query_norm.split())
    t = set(text_norm.split())
    if not q:
        return 0.0
    return len(q & t) / len(q)


# ============================================================
#   BUSCA PRINCIPAL
# ============================================================
def cerca_immagini(
    descrizione: str,
    categoria: str,
    file_indice_faiss: str,
    file_percorsi: str,
    directories_selezionate: Optional[List[str]] = None,
    offset: int = 0,
    limit: int = 100,
    prefer_ocr: bool = False
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Quando prefer_ocr=True:
      1) prioriza OCR (marca cada item: extra_ocr_match=True);
      2) inclui vizinhos por similaridade visual do top-1 OCR;
      3) preenche com CLIP texto→imagem se ainda couber;
    Caso contrário: fluxo CLIP normal.
    """
    directories_selezionate = directories_selezionate or []

    # ---------- OCR-first opcional ----------
    if prefer_ocr:
        try:
            query_norm = _norm_text(descrizione or "")
            ocr = __ocr_load()
            if not query_norm or not ocr:
                # sem OCR -> segue fluxo padrão
                return _cerca_clip_text(
                    descrizione, file_indice_faiss, file_percorsi,
                    directories_selezionate, offset, limit
                )

            # filtra OCR por diretórios ativos e pontua
            items = list(ocr.items())  # [(path, text), ...]
            mask = _filter_by_dirs([p for p, _ in items], directories_selezionate)
            scored: List[Tuple[str, float]] = []
            for (path, text), ok in zip(items, mask):
                if not ok or not os.path.exists(path):
                    continue
                sc = __ocr_score(query_norm, _norm_text(text))
                if sc > 0.0:
                    scored.append((path, sc))

            if not scored:
                return _cerca_clip_text(
                    descrizione, file_indice_faiss, file_percorsi,
                    directories_selezionate, offset, limit
                )

            # ordena por score OCR desc; paginação aplicada nos OCR
            scored.sort(key=lambda x: (-x[1], x[0]))
            ocr_total = len(scored)
            ocr_slice = scored[offset: offset + limit]

            risultati: List[Dict[str, Any]] = [{
                "percorso": p,
                "percorso_url": p,
                "score": float(sc),
                "extra_ocr_match": True
            } for (p, sc) in ocr_slice]

            # vizinhos do top-1 OCR (se couber na página)
            space_left = max(0, limit - len(risultati))
            if space_left > 0 and scored:
                top_path = scored[0][0]
                try:
                    emb_top = extrai_features_imagem(top_path)
                    vicini, _ = cerca_per_embedding(
                        emb_top, file_indice_faiss, file_percorsi,
                        directories_selezionate=directories_selezionate,
                        offset=0, limit=limit * 2  # pega um pouco mais p/ dedup
                    )
                    seen = {r["percorso"] for r in risultati}
                    for r in vicini:
                        if r["percorso"] not in seen:
                            r["extra_ocr_match"] = False
                            risultati.append(r)
                            if len(risultati) >= offset + limit:
                                break
                except Exception:
                    pass

            # corta para a janela pedida
            risultati = risultati[:limit]

            # fallback CLIP se ainda couber
            if len(risultati) < limit:
                clip_fallback, _ = _cerca_clip_text(
                    descrizione, file_indice_faiss, file_percorsi,
                    directories_selezionate, offset=0, limit=limit * 2
                )
                seen = {r["percorso"] for r in risultati}
                for r in clip_fallback:
                    if r["percorso"] not in seen:
                        risultati.append(r)
                        if len(risultati) >= limit:
                            break

            total = max(ocr_total, len(risultati))
            return risultati, total

        except Exception:
            # falha no OCR-first → fluxo padrão
            return _cerca_clip_text(
                descrizione, file_indice_faiss, file_percorsi,
                directories_selezionate, offset, limit
            )

    # ---------- fluxo padrão: CLIP texto→imagem ----------
    return _cerca_clip_text(
        descrizione, file_indice_faiss, file_percorsi,
        directories_selezionate, offset, limit
    )


# ============================================================
#   DUPLICATAS
# ============================================================
def encontrar_duplicatas(
    embeddings_file: str,
    file_percorsi: str,
    threshold_duplicates: float = 0.99,
    max_neighbors: int = 50
) -> List[List[Dict[str, Any]]]:
    """
    Agrupa imagens cujos embeddings têm similaridade (cosine/IP) >= threshold.
    Retorna lista de grupos; cada grupo é uma lista de dicts:
      {"percorso": <path>, "data_modifica": <timestamp>, "score": <float>}
    - Usa FAISS IndexFlatIP (vetores devem estar normalizados L2).
    - Faz union-find para formar grupos sem sobreposição.

    Observação: threshold 0.99 é bem alto (quase idêntico).
    """
    if not os.path.exists(embeddings_file) or not os.path.exists(file_percorsi):
        return []

    # carrega dados
    with open(embeddings_file, "rb") as f:
        X = np.load(f).astype("float32")
    paths = _load_paths(file_percorsi)

    if X.ndim != 2 or len(paths) != X.shape[0] or X.shape[0] == 0:
        return []

    # garante normalização
    faiss.normalize_L2(X)
    d = X.shape[1]

    # indexa
    index = faiss.IndexFlatIP(d)
    index.add(X)

    # vizinhos por lote para economia de memória
    n = X.shape[0]
    k = min(max_neighbors, n)
    D = np.empty((n, k), dtype="float32")
    I = np.empty((n, k), dtype="int64")

    bs = 8192 if n >= 8192 else max(256, 2 ** int(math.log2(max(64, n // 16))))
    for start in range(0, n, bs):
        end = min(n, start + bs)
        dists, idxs = index.search(X[start:end], k)
        D[start:end] = dists
        I[start:end] = idxs

    # union-find
    parent = list(range(n))
    rank = [0] * n

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    # cria arestas para pares acima do threshold (evita j<=i para não duplicar)
    for i in range(n):
        for jj in range(1, k):  # 0 é o próprio i
            j = I[i, jj]
            if j < 0:
                continue
            if j <= i:
                continue
            sim = D[i, jj]
            if sim >= threshold_duplicates:
                union(i, j)

    # agrega por componente
    groups_idx: Dict[int, List[int]] = {}
    for i in range(n):
        r = find(i)
        groups_idx.setdefault(r, []).append(i)

    # converte para estrutura esperada
    groups: List[List[Dict[str, Any]]] = []
    for comp in groups_idx.values():
        if len(comp) <= 1:
            continue  # só interessados em grupos com 2+
        g = []
        for idx in comp:
            p = paths[idx]
            try:
                mtime = os.path.getmtime(p)
            except Exception:
                mtime = 0.0
            # score opcional: max similaridade desse item com os vizinhos do grupo
            # (não é perfeito, mas útil para ordenar internamente)
            max_sc = 1.0
            try:
                # pega a linha correspondente em D/I para achar melhor par do grupo
                row = np.where(I[idx] == idx)[0]
                if row.size == 0:
                    # se índice exato não está, calcula topar rapidamente
                    max_sc = float(np.max(D[idx]))
                else:
                    max_sc = 1.0
            except Exception:
                pass
            g.append({
                "percorso": p,
                "data_modifica": mtime,
                "score": float(max_sc)
            })
        # ordena por data_modifica desc (o app.py espera isso para apagar os mais antigos)
        g.sort(key=lambda x: x["data_modifica"], reverse=True)
        groups.append(g)

    # opcional: ordenar grupos por tamanho decrescente
    groups.sort(key=lambda comp: (-len(comp), comp[0]["percorso"] if comp else ""))

    return groups
