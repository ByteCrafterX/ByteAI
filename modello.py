import os
import time
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
import pickle

# Definição das categorias com suas descrições
CATEGORIE = {
    'ABSTRACTS': ['Abstract geometric patterns, soft geo, minimal shapes, abstract art, graffiti, splash, marble, Pucci, Street Art, hand-drawn brush strokes, Menphis style, retro art, grunge, bold seamless background'],
    'ANIMAL SKIN': ['Zebra, giraffe, leopard, tiger, and cheetah seamless skins, butterfly wings, crocodile, reptile, snake, feather patterns, fish scales, fur. Mix of animal skins, hand-drawn textures, abstract animal prints'],
    'CAMOUFLAGES': ['Nature-inspired camouflage, military army green, hunting camo, colorful patterns, leaves camouflage. Digital camouflage textures and trendy camouflage designs'],
    'CHECKS': ['Glen plaid, tartan, graphic checks, madras, houndstooth, gingham, Prince of Wales, Vichy, damier pattern, clan tartans. Various classic check patterns'],
    'CONVERSATIONAL': ['Animals, astrology, beach life, stickers, city scenes, cartoons, fruit, hearts, lips, jewels, jungle, nautical, seasonal themes, holiday, sport, symbols, western themes.'],
    'DOTS': ['Pin dots, polka dots, coin dots, confetti dots, seamless dot patterns, pois designs, pointillism, rain dotty effects.'],
    'ETHNICS': ['African, Greek, Arabic, Turkish, mosaic, tribal, folk floral, Persian antique, Shibori, Azulejo, Mexican, Moroccan, Indian patterns, wax prints, batik, Ndebele, Majolica tiles.'],
    'FLOWERS': ['Botanical flowers, bouquets, ditsy, Liberty, peony, poppy, daisies, painterly, tropical, monochrome, floral borders, tapestry, roses, tulips, Sanderson patterns'],
    'FURNITURE & TAPESTRIES': ['Tapestry floral, Art Nouveau, William Morris, Damask, Liberty, Rococo, Victorian, ornamental borders, Fleur de Lis, retro style, Toile de Jouy, antique fabric patterns'],
    'ORIENTALS':['Oriental motifs, Chinoiserie wallpaper, blossom trees, botanical themes, ink painting, Chinese-style designs, traditional oriental patterns']
}

# Carregamento do modelo CLIP e do processador
modello = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processore = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Definir o dispositivo (CPU)
device = torch.device('cpu')
modello = modello.to(device)

# Função para realizar a busca de imagens
def cerca_immagini(descrizione, categoria, file_indice_faiss, file_percorsi, offset=0, limit=None):
    descrizione_completa = descrizione

    if categoria:
        if categoria in CATEGORIE:
            descrizione_completa += ' ' + ' '.join(CATEGORIE[categoria])

    # Carregamento do índice FAISS e dos caminhos das imagens
    indice_faiss = faiss.read_index(file_indice_faiss)
    with open(file_percorsi, 'rb') as f:
        percorsi_immagini = pickle.load(f)

    # Processa a descrição de pesquisa
    inputs_testo = processore(text=[descrizione_completa], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        embedding_testo = modello.get_text_features(**inputs_testo)
    embedding_testo = embedding_testo.cpu().numpy().astype('float32')

    faiss.normalize_L2(embedding_testo)

    # Define k como 4000 para obter as 4000 imagens mais similares
    k = 4000

    # Certifique-se de que k não seja maior que o número total de imagens
    total_immagini = len(percorsi_immagini)
    if k > total_immagini:
        k = total_immagini

    # Realiza a busca FAISS
    distanze, indici = indice_faiss.search(embedding_testo, k)
    similarita = distanze[0]

    # Normaliza as similaridades para o intervalo [0,1]
    similarita = (similarita + 1) / 2

    # Processa os resultados
    risultati = []
    for idx, sim in zip(indici[0], similarita):
        percorso_immagine = percorsi_immagini[idx]
        risultati.append({
            'percorso': percorso_immagine,
            'similarita': float(sim)
        })

    # Atualiza total_immagini para refletir o número de resultados obtidos
    total_immagini = len(risultati)

    # Aplica a paginação
    if limit is not None:
        risultati_paginati = risultati[offset:offset + limit]
    else:
        risultati_paginati = risultati[offset:]

    # Logs para depuração
    print(f"Offset recebido: {offset}, Limit recebido: {limit}")
    print(f"Numero totale di immagini dopo il filtraggio: {total_immagini}")
    print(f"Numero di immagini restituite: {len(risultati_paginati)}")

    return risultati_paginati, total_immagini

# Função para encontrar duplicatas
def encontrar_duplicatas(embeddings_file, file_percorsi, threshold_duplicates=0.95, threshold_variants=0.90):
    # Carrega os embeddings das imagens
    embeddings = np.load(embeddings_file)
    embeddings = embeddings.astype('float32')
    faiss.normalize_L2(embeddings)

    # Carrega os percorsi das imagens
    with open(file_percorsi, 'rb') as f:
        percorsi_immagini = pickle.load(f)

    # Cria um índice FAISS
    dimensione = embeddings.shape[1]
    indice = faiss.IndexFlatIP(dimensione)
    indice.add(embeddings)

    # Busca k vizinhos mais próximos para cada imagem
    k = 20  # Ajuste conforme necessário
    distanze, indici = indice.search(embeddings, k)

    visitati = set()
    grupos_duplicatas = []

    for i in range(len(embeddings)):
        if i in visitati:
            continue
        gruppo = []
        for j, distanza in zip(indici[i], distanze[i]):
            if i == j:
                continue
            if distanza >= threshold_duplicates:
                visitati.add(j)
                data_modifica = time.ctime(os.path.getmtime(percorsi_immagini[j]))
                gruppo.append({
                    'percorso': percorsi_immagini[j],
                    'similarita': float(distanza),
                    'data_modifica': data_modifica
                })
            elif distanza >= threshold_variants:
                # Trata como variante (opcional)
                pass
        if gruppo:
            # Adiciona a imagem base
            data_modifica = time.ctime(os.path.getmtime(percorsi_immagini[i]))
            gruppo.append({
                'percorso': percorsi_immagini[i],
                'similarita': 1.0,
                'data_modifica': data_modifica
            })
            grupos_duplicatas.append(gruppo)

    return grupos_duplicatas
