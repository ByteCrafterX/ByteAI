import os
import time
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
import pickle
import pathlib
import json

# =============================================================================
# 1. Carrega categorias e tags
# =============================================================================
def carica_categorie():
    if not os.path.exists('categorie.json'):
        # Cria um arquivo JSON vazio se não existir
        with open('categorie.json', 'w') as f:
            json.dump({}, f, ensure_ascii=False, indent=4)
        return {}
    with open('categorie.json', 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

CATEGORIE = carica_categorie()

# =============================================================================
# 2. Carrega modelo CLIP e processador
# =============================================================================
modello = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processore = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

device = torch.device('cpu')
modello = modello.to(device)

# =============================================================================
# 3. Função para extrair embeddings de uma IMAGEM
# =============================================================================
def extrai_features_imagem(caminho_imagem):
    """
    Abre a imagem, extrai embeddings usando o CLIP
    e retorna um vetor numpy normalizado.
    """
    from PIL import Image
    img = Image.open(caminho_imagem).convert('RGB')
    inputs_img = processore(images=img, return_tensors="pt").to(device)

    with torch.no_grad():
        embedding_img = modello.get_image_features(**inputs_img)

    embedding_img = embedding_img.cpu().numpy().astype('float32')
    faiss.normalize_L2(embedding_img)  # Normaliza L2 para ficar coerente com o índice
    return embedding_img

# =============================================================================
# 4. Busca de imagens via TEXTO (já existente)
# =============================================================================
def cerca_immagini(descrizione, categoria, file_indice_faiss, file_percorsi,
                   directories_selezionate=None, offset=0, limit=None):
    descrizione_completa = descrizione

    # Concatena a descrição da categoria
    if categoria:
        if categoria in CATEGORIE:
            desc_cat = CATEGORIE[categoria].get('descrizione', '')
            if desc_cat:
                descrizione_completa += ' ' + ' '.join(desc_cat.split(', '))

    # Verifica índice e percorsi
    if not os.path.exists(file_indice_faiss) or not os.path.exists(file_percorsi):
        print("Indice FAISS ou percorsi_immagini.pkl não encontrados.")
        return [], 0

    indice_faiss = faiss.read_index(file_indice_faiss)
    with open(file_percorsi, 'rb') as f:
        percorsi_immagini = pickle.load(f)

    # Extrai embedding do texto
    inputs_testo = processore(text=[descrizione_completa], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        embedding_testo = modello.get_text_features(**inputs_testo)
    embedding_testo = embedding_testo.cpu().numpy().astype('float32')
    faiss.normalize_L2(embedding_testo)

    # k é o número total de imagens no índice
    k = len(percorsi_immagini)

    # Busca FAISS
    distanze, indici = indice_faiss.search(embedding_testo, k)
    similarita = distanze[0]
    # Normaliza as similaridades para [0,1]
    similarita = (similarita + 1) / 2

    # Monta a lista de resultados
    resultados = []
    for idx, sim in zip(indici[0], similarita):
        percurso_img = percorsi_immagini[idx]
        percurso_url = pathlib.PurePath(percurso_img).as_posix().lstrip('/')
        resultados.append({
            'percorso': percurso_img,
            'percorso_url': percurso_url,
            'similarita': float(sim)
        })

    # Filtra pelas diretórias selecionadas, se houver
    if directories_selezionate:
        dirs_set = set(directories_selezionate)
        resultados = [img for img in resultados
                      if any(img['percorso'].startswith(d) for d in dirs_set)]

    total_immagini = len(resultados)

    # Paginação
    if limit is not None:
        resultados_paginados = resultados[offset:offset + limit]
    else:
        resultados_paginados = resultados[offset:]

    return resultados_paginados, total_immagini

# =============================================================================
# 5. Busca de imagens via EMBEDDING de IMAGEM (NOVA)
# =============================================================================
def cerca_per_embedding(embedding_img, file_indice_faiss, file_percorsi,
                        directories_selezionate=None, offset=0, limit=None):
    """
    Faz busca FAISS usando um embedding de IMAGEM (já normalizado).
    Retorna (resultados, total_resultados).
    """
    if not os.path.exists(file_indice_faiss) or not os.path.exists(file_percorsi):
        print("Indice FAISS ou percorsi_immagini.pkl não encontrados.")
        return [], 0

    indice_faiss = faiss.read_index(file_indice_faiss)
    with open(file_percorsi, 'rb') as f:
        percorsi_immagini = pickle.load(f)

    k = len(percorsi_immagini)

    # Realiza a busca FAISS com o embedding da imagem
    distanze, indici = indice_faiss.search(embedding_img, k)
    similarita = distanze[0]
    # Normaliza as similaridades para [0,1]
    similarita = (similarita + 1) / 2

    resultados = []
    for idx, sim in zip(indici[0], similarita):
        percurso_img = percorsi_immagini[idx]
        percurso_url = pathlib.PurePath(percurso_img).as_posix().lstrip('/')
        resultados.append({
            'percorso': percurso_img,
            'percorso_url': percurso_url,
            'similarita': float(sim)
        })

    # Filtra por diretórios selecionados, se houver
    if directories_selezionate:
        dirs_set = set(directories_selezionate)
        resultados = [img for img in resultados
                      if any(img['percorso'].startswith(d) for d in dirs_set)]

    total_immagini = len(resultados)

    # Paginação
    if limit is not None:
        resultados_paginados = resultados[offset:offset + limit]
    else:
        resultados_paginados = resultados[offset:]

    return resultados_paginados, total_immagini

# =============================================================================
# 6. Encontrar duplicatas
# =============================================================================
def encontrar_duplicatas(embeddings_file, file_percorsi,
                         threshold_duplicates=0.95, threshold_variants=0.90):
    # Carrega os embeddings
    if not os.path.exists(embeddings_file) or not os.path.exists(file_percorsi):
        print("Arquivo de embeddings ou percorsi_immagini.pkl não encontrado.")
        return []

    embeddings = np.load(embeddings_file).astype('float32')
    faiss.normalize_L2(embeddings)

    with open(file_percorsi, 'rb') as f:
        percorsi_immagini = pickle.load(f)

    # Filtra apenas as imagens que existem
    percorsi_ok = []
    emb_ok = []
    for idx, p in enumerate(percorsi_immagini):
        if os.path.exists(p):
            percorsi_ok.append(p)
            emb_ok.append(embeddings[idx])
        else:
            print(f"Imagem não encontrada, ignorada: {p}")

    if not emb_ok:
        return []

    emb_ok = np.array(emb_ok)
    dimensione = emb_ok.shape[1]
    indice = faiss.IndexFlatIP(dimensione)
    indice.add(emb_ok)

    k = 20  # Ajuste conforme necessário
    distanze, indici = indice.search(emb_ok, k)

    visitados = set()
    grupos_duplicatas = []

    for i in range(len(emb_ok)):
        if i in visitados:
            continue
        grupo = []
        for j, dist_ij in zip(indici[i], distanze[i]):
            if i == j:
                continue
            if dist_ij >= threshold_duplicates:
                if j in visitados:
                    continue
                visitados.add(j)
                data_mod = time.ctime(os.path.getmtime(percorsi_ok[j]))
                grupo.append({
                    'percorso': percorsi_ok[j],
                    'percorso_relativo': percorsi_ok[j],
                    'similarita': float(dist_ij),
                    'data_modifica': data_mod
                })
            elif dist_ij >= threshold_variants:
                # Lógica para variantes, se desejar
                pass
        if grupo:
            data_mod = time.ctime(os.path.getmtime(percorsi_ok[i]))
            grupo.append({
                'percorso': percorsi_ok[i],
                'percorso_relativo': percorsi_ok[i],
                'similarita': 1.0,
                'data_modifica': data_mod
            })
            grupos_duplicatas.append(grupo)

    return grupos_duplicatas
