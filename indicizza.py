import os
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import faiss
import pickle
from progress import indicizzazione_progress, progress_lock

def indicizza_immagini(directory_immagini, file_indice_faiss, file_percorsi):
    # Carrega o modelo e o processador
    modello = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processore = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Define o dispositivo (CPU)
    device = torch.device('cpu')
    modello = modello.to(device)

    embedding_immagini = []
    percorsi_immagini = []

    immagini_da_indicizzare = []
    for root, dirs, files in os.walk(directory_immagini):
        # Ignora diretórios ocultos
        dirs[:] = [d for d in dirs if not d.startswith('.')]

        for nome_file in files:
            # Ignora arquivos ocultos e garante que o arquivo seja uma imagem válida
            if nome_file.startswith('.') or '@thumb' in nome_file:
                continue

            # Verifica extensões de imagem válidas
            if nome_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                percorso_file = os.path.join(root, nome_file)
                immagini_da_indicizzare.append(percorso_file)

    totale_immagini = len(immagini_da_indicizzare)
    if totale_immagini == 0:
        with progress_lock:
            indicizzazione_progress['log'].append("Nessuna immagine trovata per l'indicizzazione.")
            indicizzazione_progress['percentuale'] = 100
        return

    with progress_lock:
        indicizzazione_progress['log'].append(f"Inizio indicizzazione di {totale_immagini} immagini...")
        indicizzazione_progress['percentuale'] = 0

    batch_size = 32  # Ajuste conforme a memória disponível
    for idx in range(0, totale_immagini, batch_size):
        batch_files = immagini_da_indicizzare[idx:idx + batch_size]
        images = []
        for f in batch_files:
            try:
                with Image.open(f).convert('RGB') as img:
                    images.append(img.copy())
                percorsi_immagini.append(os.path.abspath(f))
            except Exception as e:
                with progress_lock:
                    indicizzazione_progress['log'].append(f"Errore nell'apertura dell'immagine {f}: {e}")
                continue

        if not images:
            continue

        inputs = processore(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            embeddings = modello.get_image_features(**inputs)
        embeddings = embeddings.cpu().numpy()
        embedding_immagini.append(embeddings)

        with progress_lock:
            percentuale = int((idx + len(batch_files)) / totale_immagini * 100)
            indicizzazione_progress['percentuale'] = percentuale
            indicizzazione_progress['log'].append(
                f"Indicizzate {idx + len(batch_files)} su {totale_immagini} immagini.")

    if embedding_immagini:
        embedding_array = np.vstack(embedding_immagini).astype('float32')
        faiss.normalize_L2(embedding_array)

        with open('embeddings_immagini.npy', 'wb') as f:
            np.save(f, embedding_array)

        dimensione = embedding_array.shape[1]
        indice_faiss = faiss.IndexFlatIP(dimensione)
        indice_faiss.add(embedding_array)

        faiss.write_index(indice_faiss, file_indice_faiss)

        with open(file_percorsi, 'wb') as f:
            pickle.dump(percorsi_immagini, f)

        with progress_lock:
            indicizzazione_progress['log'].append("Indicizzazione completata con successo.")
            indicizzazione_progress['percentuale'] = 100
    else:
        with progress_lock:
            indicizzazione_progress['log'].append("Nessuna immagine è stata indicizzata.")
            indicizzazione_progress['percentuale'] = 100
