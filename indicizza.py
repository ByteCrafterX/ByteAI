# indizza.py – indexação + OCR (na RAIZ) com limpeza de “fantasmas”
import os, pickle, json, re
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import faiss

from progress import indicizzazione_progress, progress_lock
from ocr_utils import extrai_texto_e_indexa  # extração; leitura/escrita do OCR faremos localmente (RAIZ)

# ===== OCR na RAIZ =====
OCR_META_PATH = "ocr_metadata.json"

def _ocr_root_load() -> dict:
    """Lê o JSON de OCR na RAIZ; cria vazio se não existir."""
    if not os.path.exists(OCR_META_PATH):
        try:
            with open(OCR_META_PATH, "w", encoding="utf-8") as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
        except Exception:
            return {}
        return {}
    try:
        with open(OCR_META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _ocr_root_save(d: dict):
    """Salva OCR na RAIZ de forma atômica."""
    tmp = OCR_META_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    os.replace(tmp, OCR_META_PATH)

def _ocr_root_clean_orphans(directories_immagini: list[str]) -> int:
    """
    Remove do OCR:
      - caminhos inexistentes em disco
      - caminhos fora das diretivas ativas
    Retorna a QTD removida.
    """
    meta = _ocr_root_load()
    if not meta:
        return 0

    dirs_norm = [os.path.normpath(x) for x in (directories_immagini or [])]
    def _in_active_dirs(pth: str) -> bool:
        if not dirs_norm:
            return True
        p = os.path.normpath(pth)
        return any(p.startswith(dn) for dn in dirs_norm)

    removed = 0
    for k in list(meta.keys()):
        if (not os.path.exists(k)) or (not _in_active_dirs(k)):
            meta.pop(k, None)
            removed += 1
    if removed:
        _ocr_root_save(meta)
    return removed


from ocr_utils import extrai_texto_e_indexa, carrega_ocr_metadata, salva_ocr_metadata  # (compat: não usados aqui)
from ocr_utils import extrai_texto_e_indexa as _unused__extrai_texto_e_indexa  # só p/ evitar linter reclamar


def indicizza_immagini(directories_immagini, file_indice_faiss,
                       file_percorsi, interrompi_evento):
    """
    Indexa incrementalmente:
      • embeddings CLIP (faiss)
      • texto OCR (ocr_metadata.json, na RAIZ)
      • remove entradas órfãs (embeddings e OCR)
    """

    # ───── 0. modelo CLIP ─────
    modello    = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processore = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device     = torch.device("cpu")
    modello    = modello.to(device)
    if modello.dtype == torch.float16:          # evita overflow fp16→cpu
        modello = modello.to(torch.float32)

    embedding_immagini: list[np.ndarray] = []
    percorsi_immagini:  list[str]        = []

    # ───── 1. índice existente ─────
    if os.path.exists("embeddings_immagini.npy") and os.path.exists(file_percorsi):
        with open("embeddings_immagini.npy", "rb") as f:
            existing_embeddings = np.load(f)
        with open(file_percorsi, "rb") as f:
            percorsi_immagini = pickle.load(f)
        embedding_immagini = [existing_embeddings]
    else:
        with progress_lock:
            indicizzazione_progress["log"].append(
                "[indicizza] Nenhum índice anterior — iniciando do zero.")

    # ───── 2. remove fantasmas ─────
    if embedding_immagini:
        array_completo = np.vstack(embedding_immagini).astype("float32")
        novos_percursos, novos_embeddings, cont_removidos = [], [], 0

        meta_ocr = _ocr_root_load()  # OCR na RAIZ

        for idx, caminho in enumerate(percorsi_immagini):
            if interrompi_evento.is_set():
                with progress_lock:
                    indicizzazione_progress["log"].append(
                        "[indicizza] Interrompido na limpeza de órfãos.")
                return

            if os.path.exists(caminho) and any(
                os.path.normpath(caminho).startswith(os.path.normpath(d))
                for d in directories_immagini
            ):
                novos_percursos.append(caminho)
                novos_embeddings.append(array_completo[idx])
            else:
                cont_removidos += 1
                # remove também do OCR
                if caminho in meta_ocr:
                    meta_ocr.pop(caminho, None)

        percorsi_immagini  = novos_percursos
        embedding_immagini = [np.vstack(novos_embeddings)] if novos_embeddings else []
        _ocr_root_save(meta_ocr)  # persiste OCR limpo

        if cont_removidos:
            with progress_lock:
                indicizzazione_progress["log"].append(
                    f"[indicizza] Removidos {cont_removidos} caminhos obsoletos (e OCR correspondente).")

    # ───── 2.b limpeza complementar de OCR (independente) ─────
    removed_ocr = _ocr_root_clean_orphans(directories_immagini)
    if removed_ocr:
        with progress_lock:
            indicizzazione_progress["log"].append(
                f"[indicizza] OCR: removidas {removed_ocr} entradas inválidas.")

    # ───── 3. novas imagens ─────
    imagens_da_indicizzare = []
    for directory_immagini in directories_immagini:
        for root, dirs, files in os.walk(directory_immagini):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for nome_file in files:
                if interrompi_evento.is_set():
                    with progress_lock:
                        indicizzazione_progress["log"].append(
                            "[indicizza] Indexação interrompida pelo usuário.")
                    return
                if nome_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    p = os.path.join(root, nome_file)
                    if p not in percorsi_immagini:
                        imagens_da_indicizzare.append(p)

    total_novas = len(imagens_da_indicizzare)
    if total_novas == 0 and not embedding_immagini:
        _limpar_indices(file_indice_faiss, file_percorsi)
        return
    if total_novas == 0:
        _salvar_indice(embedding_immagini, percorsi_immagini,
                       file_indice_faiss, file_percorsi)
        return

    with progress_lock:
        indicizzazione_progress["log"].append(
            f"[indicizza] Encontradas {total_novas} novas imagens.")
        indicizzazione_progress["percentuale"] = 0

    # ───── 4. embeddings + OCR ─────
    batch_size = 32
    lista_novos = []

    for idx in range(0, total_novas, batch_size):
        if interrompi_evento.is_set():
            with progress_lock:
                indicizzazione_progress["log"].append(
                    "[indicizza] Interrompido pelo usuário.")
            return

        batch_files = imagens_da_indicizzare[idx: idx + batch_size]
        images_pil  = []
        for f_ in batch_files:
            try:
                with Image.open(f_).convert("RGB") as im:
                    images_pil.append(im.copy())
                percorsi_immagini.append(os.path.abspath(f_))

                # --- OCR (extrai e registra na RAIZ) ---
                extrai_texto_e_indexa(f_)  # util salva no arquivo
                meta = _ocr_root_load()
                ocr_txt = meta.get(f_, "")
                char_info = f"{len(ocr_txt.strip())} char" if ocr_txt.strip() else "nenhum texto"
                with progress_lock:
                    indicizzazione_progress["log"].append(
                        f"[OCR] {os.path.basename(f_)} → {char_info}")

            except Exception as e:
                with progress_lock:
                    indicizzazione_progress["log"].append(
                        f"[ERRO] abrindo {f_}: {e}")
                continue

        if not images_pil:
            continue

        inputs = processore(images=images_pil,
                            return_tensors="pt",
                            padding=True).to(device)
        with torch.no_grad():
            embeddings_batch = modello.get_image_features(**inputs)
        lista_novos.append(embeddings_batch.cpu().numpy())

        with progress_lock:
            pct = int((idx + len(batch_files)) / total_novas * 100)
            indicizzazione_progress["percentuale"] = pct

    # ───── 5. salva índice ─────
    if lista_novos:
        novos_array = np.vstack(lista_novos).astype("float32")
        final_array = (
            np.concatenate((np.vstack(embedding_immagini).astype("float32"),
                           novos_array), axis=0)
            if embedding_immagini else novos_array
        )
    else:
        final_array = (np.vstack(embedding_immagini).astype("float32")
                       if embedding_immagini else np.zeros((0, 512), "float32"))

    if final_array.shape[0] == 0:
        _limpar_indices(file_indice_faiss, file_percorsi)
    else:
        faiss.normalize_L2(final_array)
        idx_faiss = faiss.IndexFlatIP(final_array.shape[1])
        idx_faiss.add(final_array)
        faiss.write_index(idx_faiss, file_indice_faiss)

        with open("embeddings_immagini.npy", "wb") as f:
            np.save(f, final_array)
        with open(file_percorsi, "wb") as f:
            pickle.dump(percorsi_immagini, f)

        # 🔹 mantém os diretórios no config.json sem apagar outras chaves
        from config_utils import save_config
        save_config({
            "directories_indicizzate": directories_immagini
        })

        with progress_lock:
            indicizzazione_progress["log"].append(
                f"[indicizza] Concluído — {final_array.shape[0]} imagens no índice.")
            indicizzazione_progress["percentuale"] = 100


# ───────── auxiliares ─────────
def _limpar_indices(file_indice_faiss, file_percorsi):
    for p in ("embeddings_immagini.npy", file_indice_faiss, file_percorsi):
        if os.path.exists(p):
            os.remove(p)
    with progress_lock:
        indicizzazione_progress["log"].append("Nenhuma imagem — índice limpo.")
        indicizzazione_progress["percentuale"] = 100


def _salvar_indice(embedding_immagini, percorsi_immagini,
                   file_indice_faiss, file_percorsi):
    if not embedding_immagini:
        _limpar_indices(file_indice_faiss, file_percorsi)
        return
    final_array = np.vstack(embedding_immagini).astype("float32")
    if final_array.size == 0:
        _limpar_indices(file_indice_faiss, file_percorsi)
        return
    faiss.normalize_L2(final_array)
    idx = faiss.IndexFlatIP(final_array.shape[1])
    idx.add(final_array)
    faiss.write_index(idx, file_indice_faiss)
    with open("embeddings_immagini.npy", "wb") as f:
        np.save(f, final_array)
    with open(file_percorsi, "wb") as f:
        pickle.dump(percorsi_immagini, f)
