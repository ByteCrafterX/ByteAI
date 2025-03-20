import os
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import faiss
import pickle

# Importando para atualizar logs e percentuais
from progress import indicizzazione_progress, progress_lock

def indicizza_immagini(directories_immagini, file_indice_faiss, file_percorsi, interrompi_evento):
    """
    Indexa (incremental) as novas imagens E remove do índice
    as que não existem mais no disco (evitando “arquivos fantasmas”).
    """

    # Carrega modelo e processador CLIP
    modello = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processore = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = torch.device('cpu')
    modello = modello.to(device)

    embedding_immagini = []
    percorsi_immagini = []

    # 1) Carrega embeddings e percursos existentes, se disponíveis
    if os.path.exists('embeddings_immagini.npy') and os.path.exists(file_percorsi):
        with open('embeddings_immagini.npy', 'rb') as f:
            existing_embeddings = np.load(f)
        with open(file_percorsi, 'rb') as f:
            percorsi_immagini = pickle.load(f)
        embedding_immagini = [existing_embeddings]
    else:
        with progress_lock:
            indicizzazione_progress['log'].append("[indicizza] Nenhum índice anterior. Iniciando do zero...")

    # 2) Remover do índice caminhos que não existem mais
    if embedding_immagini:
        from numpy import vstack
        array_completo = vstack(embedding_immagini).astype("float32")
        novos_percursos = []
        novos_embeddings = []
        cont_removidos = 0

        for idx, caminho in enumerate(percorsi_immagini):
            if interrompi_evento.is_set():
                with progress_lock:
                    indicizzazione_progress['log'].append("[indicizza] Indexação interrompida durante remoção de obsoletos.")
                return

            # Se o caminho ainda existe e está numa das dirs monitoradas, mantemos
            if os.path.exists(caminho) and any(caminho.startswith(d) for d in directories_immagini):
                novos_percursos.append(caminho)
                novos_embeddings.append(array_completo[idx])
            else:
                cont_removidos += 1

        percorsi_immagini = novos_percursos
        if novos_embeddings:
            embedding_immagini = [np.vstack(novos_embeddings)]
        else:
            embedding_immagini = []

        if cont_removidos > 0:
            with progress_lock:
                indicizzazione_progress['log'].append(f"[indicizza] Removidos {cont_removidos} arquivos inexistentes do índice.")

    # 3) Ver quais imagens ainda não estão no índice
    imagens_da_indicizzare = []
    for directory_immagini in directories_immagini:
        for root, dirs, files in os.walk(directory_immagini):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for nome_file in files:
                if interrompi_evento.is_set():
                    with progress_lock:
                        indicizzazione_progress['log'].append("Indicizzazione interrotta dal'utente.")
                    return
                if nome_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    percurso_file = os.path.join(root, nome_file)
                    # Se não está no percorsi_immagini, é imagem nova
                    if percurso_file not in percorsi_immagini:
                        imagens_da_indicizzare.append(percurso_file)

    total_novas = len(imagens_da_indicizzare)
    if total_novas == 0 and not embedding_immagini:
        # Significa que não sobrou nada no disco
        with progress_lock:
            indicizzazione_progress['log'].append("[indicizza] Nenhuma imagem no disco. Limpando índice.")
            indicizzazione_progress['percentuale'] = 100

        if os.path.exists('embeddings_immagini.npy'):
            os.remove('embeddings_immagini.npy')
        if os.path.exists(file_indice_faiss):
            os.remove(file_indice_faiss)
        if os.path.exists(file_percorsi):
            os.remove(file_percorsi)
        return

    if total_novas == 0:
        # Sem imagens novas, apenas atualiza o índice (regrava removidos, se houve)
        with progress_lock:
            indicizzazione_progress['log'].append("[indicizza] Sem novas imagens. Salvando índice após remoção de obsoletos.")
        _salvar_indice(embedding_immagini, percorsi_immagini, file_indice_faiss, file_percorsi)
        return

    # 4) Calcular embedding das novas imagens
    with progress_lock:
        indicizzazione_progress['log'].append(f"[indicizza] Encontradas {total_novas} novas imagens para indexar.")
        indicizzazione_progress['percentuale'] = 0

    from torch import no_grad
    batch_size = 32
    lista_novos = []

    for idx in range(0, total_novas, batch_size):
        if interrompi_evento.is_set():
            with progress_lock:
                indicizzazione_progress['log'].append("Indicizzazione interrotta dall'utente.")
            return

        batch_files = imagens_da_indicizzare[idx:idx + batch_size]
        images_pil = []
        for f_ in batch_files:
            try:
                with Image.open(f_).convert('RGB') as img:
                    images_pil.append(img.copy())
                percorsi_immagini.append(os.path.abspath(f_))
            except Exception as e:
                with progress_lock:
                    indicizzazione_progress['log'].append(f"Errore abrindo {f_}: {e}")
                continue

        if not images_pil:
            continue

        inputs = processore(images=images_pil, return_tensors="pt", padding=True).to(device)
        with no_grad():
            embeddings_batch = modello.get_image_features(**inputs)
        lista_novos.append(embeddings_batch.cpu().numpy())

        # Atualiza progresso
        with progress_lock:
            percentuale = int((idx + len(batch_files)) / total_novas * 100)
            indicizzazione_progress['percentuale'] = percentuale
            indicizzazione_progress['log'].append(f"Indexadas {idx + len(batch_files)} de {total_novas} novas imagens.")

    # Concatena embeddings antigos e novos
    if lista_novos:
        novos_array = np.vstack(lista_novos).astype("float32")
        if embedding_immagini:
            old_array = np.vstack(embedding_immagini).astype("float32")
            final_array = np.concatenate((old_array, novos_array), axis=0)
        else:
            final_array = novos_array
    else:
        # Não houve novas embeddings; ficamos só com as antigas
        if embedding_immagini:
            final_array = np.vstack(embedding_immagini).astype("float32")
        else:
            final_array = np.zeros((0,512), dtype="float32")

    # 5) Salvar o índice final
    if final_array.shape[0] == 0:
        # Se no final não restou nenhuma imagem
        with progress_lock:
            indicizzazione_progress['log'].append("Nenhuma imagem restou. Limpando índice.")
            indicizzazione_progress['percentuale'] = 100
        if os.path.exists('embeddings_immagini.npy'):
            os.remove('embeddings_immagini.npy')
        if os.path.exists(file_indice_faiss):
            os.remove(file_indice_faiss)
        if os.path.exists(file_percorsi):
            os.remove(file_percorsi)
    else:
        faiss.normalize_L2(final_array)
        dimensione = final_array.shape[1]
        indice_faiss = faiss.IndexFlatIP(dimensione)
        indice_faiss.add(final_array)
        faiss.write_index(indice_faiss, file_indice_faiss)

        with open('embeddings_immagini.npy', 'wb') as f:
            np.save(f, final_array)
        with open(file_percorsi, 'wb') as f:
            pickle.dump(percorsi_immagini, f)

        with progress_lock:
            indicizzazione_progress['log'].append(f"Indexação finalizada. Total de {final_array.shape[0]} imagens.")
            indicizzazione_progress['percentuale'] = 100


def _salvar_indice(embedding_immagini, percorsi_immagini, file_indice_faiss, file_percorsi):
    """
    Função auxiliar para regravar o índice caso não haja novas imagens,
    mas tenhamos removido entradas obsoletas.
    """
    import faiss
    import numpy as np

    if embedding_immagini:
        final_array = np.vstack(embedding_immagini).astype("float32")
        if final_array.shape[0] > 0:
            faiss.normalize_L2(final_array)
            dimensione = final_array.shape[1]
            indice_faiss = faiss.IndexFlatIP(dimensione)
            indice_faiss.add(final_array)
            faiss.write_index(indice_faiss, file_indice_faiss)

            with open('embeddings_immagini.npy', 'wb') as f:
                np.save(f, final_array)
            with open(file_percorsi, 'wb') as f:
                pickle.dump(percorsi_immagini, f)
        else:
            # Se não sobrou nada
            if os.path.exists('embeddings_immagini.npy'):
                os.remove('embeddings_immagini.npy')
            if os.path.exists(file_indice_faiss):
                os.remove(file_indice_faiss)
            if os.path.exists(file_percorsi):
                os.remove(file_percorsi)
    else:
        # embedding_immagini vazio => remove tudo
        if os.path.exists('embeddings_immagini.npy'):
            os.remove('embeddings_immagini.npy')
        if os.path.exists(file_indice_faiss):
            os.remove(file_indice_faiss)
        if os.path.exists(file_percorsi):
            os.remove(file_percorsi)
