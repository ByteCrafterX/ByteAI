# indizza.py – indexação + OCR
import os, pickle, json, re
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import faiss

from progress import indicizzazione_progress, progress_lock
from ocr_utils import extrai_texto_e_indexa, carrega_ocr_metadata, salva_ocr_metadata


def indicizza_immagini(directories_immagini, file_indice_faiss,
                       file_percorsi, interrompi_evento):
    """
    Indexa incrementalmente:
      • embeddings CLIP (faiss)
      • texto OCR (ocr_metadata.json)
      • remove entradas órfãs
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

        meta_ocr = carrega_ocr_metadata()

        for idx, caminho in enumerate(percorsi_immagini):
            if interrompi_evento.is_set():
                with progress_lock:
                    indicizzazione_progress["log"].append(
                        "[indicizza] Interrompido na limpeza de órfãos.")
                return

            if os.path.exists(caminho) and any(
                caminho.startswith(d) for d in directories_immagini
            ):
                novos_percursos.append(caminho)
                novos_embeddings.append(array_completo[idx])
            else:
                cont_removidos += 1
                meta_ocr.pop(caminho, None)

        percorsi_immagini  = novos_percursos
        embedding_immagini = [np.vstack(novos_embeddings)] if novos_embeddings else []
        salva_ocr_metadata(meta_ocr)

        if cont_removidos:
            with progress_lock:
                indicizzazione_progress["log"].append(
                    f"[indicizza] Removidos {cont_removidos} caminhos obsoletos.")

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

                # --- OCR ---
                extrai_texto_e_indexa(f_)
                ocr_txt = carrega_ocr_metadata().get(f_, "")
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
