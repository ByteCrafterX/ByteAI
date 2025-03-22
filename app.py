import logging
from logging.handlers import TimedRotatingFileHandler
import os
import threading
import json
import subprocess
import pickle
import numpy as np
import faiss
from PIL import Image
from io import BytesIO
import zipfile
import tempfile
import shutil
import string
import time
import random
# ========================= CONFIGURAÇÃO DO LOGGER =========================
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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

# Ativa debug global (APScheduler, etc.)
logging.basicConfig(level=logging.DEBUG)

from flask import (
    Flask, render_template, request, send_file, redirect, url_for,
    flash, jsonify, abort, session
)

# ========================= IMPORTS DO SEU PROJETO =========================
from modello import (
    cerca_immagini,
    encontrar_duplicatas,
    extrai_features_imagem,
    cerca_per_embedding
)
from indicizza import indicizza_immagini
from progress import indicizzazione_progress, progress_lock

# =============== IMPORT DO APSCHEDULER ===============
from apscheduler.schedulers.background import BackgroundScheduler
# ======================================================

app = Flask(__name__)
app.secret_key = 'chiave_segreta_per_flash_message'

# ========================= ARQUIVOS E CONFIGS =========================
file_indice_faiss = 'indice_faiss.index'
file_percorsi = 'percorsi_immagini.pkl'
embeddings_file = 'embeddings_immagini.npy'
config_file = 'config.json'
categorie_file = 'categorie.json'

# ==================== VARIÁVEIS GLOBAIS ====================
indicizzazione_thread = None
indicizzazione_interrompida = threading.Event()

directories_indicizzate = []
# ==== VARIÁVEIS GLOBAIS DE PROGRESSO DO TREINO LORA ====
lora_lock = threading.Lock()
lora_progress = {
    "running": False,
    "percent": 0,
    "log": [],
    "completed": False
}
lora_thread = None

try:
    def carica_configurazione():
        """
        Carrega a lista de directories_indicizzate no formato:
        [
          {"path": "/caminho/absoluto", "nome": "ApelidoOpcional"},
          ...
        ]
        Se o config estiver no formato antigo (lista de strings),
        converte para lista de dicionários com nome="".
        """
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
                # Formato antigo (apenas path)
                new_list.append({"path": d, "nome": ""})
            elif isinstance(d, dict):
                path_ = d.get("path", "")
                nome_ = d.get("nome", "")
                new_list.append({"path": path_, "nome": nome_})
        return new_list

    def salva_configurazione(directories_indicizzate_local):
        """
        Salva em config.json -> 'directories_indicizzate': lista de dict
        (E mantém intacto qualquer outra chave do config.json, ex: generative_dirs)
        """
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

    directories_indicizzate = carica_configurazione()
except Exception as e:
    logger.error("[ERROR] Carregando configurações: %s", e)
    directories_indicizzate = []


# ========== NOVO: Carregar/Salvar info de diretórios para geração (generative_dirs) ==========
def carica_configurazione_generativa():
    """
    Retorna algo como:
    {
      "path_dir1": true/false,
      "path_dir2": true/false
    }
    indicando se cada diretório está habilitado para geração.
    """
    if not os.path.exists(config_file):
        return {}
    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)
            return config_data.get("generative_dirs", {})
    except:
        return {}

def salva_configurazione_generativa(generative_data):
    """
    Salva no config.json em 'generative_dirs'.
    Mantém intacto o resto do config.
    """
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


# Funções para carregar e salvar usuários no mesmo config.json
def carica_utenti():
    """Carrega a lista de usuários do config.json, retorna lista de dicionários."""
    if not os.path.exists(config_file):
        return []
    try:
        with open(config_file, 'r') as f:
            data = json.load(f)
        return data.get('users', [])
    except:
        return []

def salva_utenti(users_list):
    """Salva a lista de usuários em config.json (em 'users')."""
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
    """
    Garante que o usuário 'chickellero' (senha 'chickellero') exista no config.json.
    Nome: 'Marco'
    """
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

schedule_lock = threading.Lock()
schedule_data = {"days": [], "hour": "00:00"}

conversion_thread = None
conversion_lock = threading.Lock()
conversion_status = {
    'percentuale': 0,
    'log': [],
    'completed': False,
    'total_files': 0,
    'processed_files': 0
}

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

# ======================================================
#   DECORATOR PARA EXIGIR LOGIN NAS ROTAS
# ======================================================
from functools import wraps

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

# ======================================================
#   ROTA PRINCIPAL (INDEX) COM POSSÍVEL LOGIN
# ======================================================
@app.route('/')
def index():
    """
    Se usuário estiver logado, mostra o formulário de pesquisa de imagens.
    Se não estiver logado, mostra o overlay de login.
    """
    return render_template(
        'index.html',
        directories_indicizzate=directories_indicizzate,
        categorie=categorie
    )

# ======================================================
#   ROTA PARA PROCESSAR O LOGIN
# ======================================================
@app.route('/fazer_login', methods=['POST'])
def fazer_login():
    username = request.form.get('username')
    password = request.form.get('password')

    # Carrega lista de usuários do config
    users = carica_utenti()
    for user in users:
        if user['username'] == username and user['password'] == password:
            session['logged_in'] = True
            session['username'] = username
            flash('Accesso effettuato con successo!', 'success')
            return redirect(url_for('index'))

    flash('Nome utente o password non validi!', 'error')
    return redirect(url_for('index'))

# ======================================================
#   ROTA PARA LOGOUT
# ======================================================
@app.route('/logout')
def logout():
    session.clear()
    flash('Logout effettuato.', 'info')
    return redirect(url_for('index'))

# ======================================================
#   PÁGINA DE ABOUT
# ======================================================
@app.route('/about')
@login_required
def about():
    return render_template('about.html')

# ======================================================
#   PÁGINA DE CONFIGURAÇÕES (Impostazioni)
#   (Agora sem a parte de "indicizzazione")
# ======================================================
@app.route('/configurazioni')
@login_required
def configurazioni():
    return render_template('config.html', categorie=categorie)

@app.route('/utilidades')
@login_required
def utilidades():
    return render_template('utilidades.html')

# ======================================================
#   NOVA PÁGINA DE INDICIZZAZIONE
# ======================================================
@app.route('/indicizzazione')
@login_required
def indicizzazione():
    """
    Exibe a página que gerencia (a) adicionar diretórios,
    (b) interromper indexação, (c) reindexar, (d) remover diretórios,
    e mostra progresso. Também permite renomear diretórios.
    """
    return render_template(
        'indicizzazione.html',
        directories_indicizzate=directories_indicizzate
    )

# ======================================================
#   ROTA PARA ADICIONAR NOVA DIRETORIA E INDICIZZAR
# ======================================================
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

    # Verifica se já existe
    for d in directories_indicizzate:
        if d['path'] == directory_input:
            flash('La directory è già stata indicizzata.', 'info config')
            return redirect(url_for('indicizzazione'))

    # Verifica se outra indexação está em curso
    if indicizzazione_thread and indicizzazione_thread.is_alive():
        flash('Indicizzazione già in corso.', 'error config')
        return redirect(url_for('indicizzazione'))

    # Seta evento para false (não interrompida)
    indicizzazione_interrompida.clear()

    # Adiciona no array e salva config
    directories_indicizzate.append({"path": directory_input, "nome": nome_input})
    salva_configurazione(directories_indicizzate)

    with progress_lock:
        indicizzazione_progress['percentuale'] = 0
        indicizzazione_progress['log'] = []

    def run_indicizzazione():
        try:
            logger.debug("Iniciando indicizza_immagini nas pastas: %s", directories_indicizzate)
            # Monta lista de paths (strings) a partir do array
            lista_paths = [d["path"] for d in directories_indicizzate]

            indicizza_immagini(lista_paths,
                               file_indice_faiss,
                               file_percorsi,
                               indicizzazione_interrompida)
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

# ======================================================
#   ROTA PARA INTERROMPER INDEXAÇÃO
# ======================================================
@app.route('/interrompi_indicizzazione', methods=['POST'])
@login_required
def interrompi_indicizzazione_route():
    logger.debug("Interrompendo indicizzazione...")
    indicizzazione_interrompida.set()
    flash('Indicizzazione interrotta.', 'info config')
    return redirect(url_for('indicizzazione'))

# ======================================================
#   ROTA PARA CONSULTAR STATUS DA INDEXAÇÃO (JSON)
# ======================================================
@app.route('/stato_indicizzazione')
@login_required
def stato_indicizzazione():
    with progress_lock:
        progress_data = indicizzazione_progress.copy()
    return jsonify(progress_data)

# ======================================================
#   ROTA PARA REINDICIZZA TUDO
# ======================================================
@app.route('/reindicizza_tutto', methods=['POST'])
@login_required
def reindicizza_tutto():
    global indicizzazione_thread
    global directories_indicizzate, indicizzazione_interrompida

    logger.debug("reindicizza_tutto chamado.")
    # Interrompe a indexação se estiver rodando
    indicizzazione_interrompida.set()
    if indicizzazione_thread and indicizzazione_thread.is_alive():
        indicizzazione_thread.join()

    # Remove os arquivos de índice
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
        from indicizza import indicizza_immagini
        try:
            logger.debug("Iniciando reindicizzazione TUDO para dirs: %s", directories_indicizzate)
            lista_paths = [d["path"] for d in directories_indicizzate]

            indicizza_immagini(
                lista_paths,
                "indice_faiss.index",
                "percorsi_immagini.pkl",
                interrompi_evento=indicizzazione_interrompida
            )
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

# ======================================================
#   ROTA PARA ATUALIZAR O NOME (apelido) DE UMA DIRECTORY
# ======================================================
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

# ======================================================
#   ROTA PARA REMOVER UMA DIRECTORY
# ======================================================
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

# ======================================================
#   PESQUISA DE IMAGENS (por texto OU upload)
# ======================================================
@app.route('/ricerca', methods=['POST'])
@login_required
def ricerca():
    global directories_indicizzate, categorie
    image_file = request.files.get('image_file')

    if image_file and image_file.filename != '':
        # BUSCA POR IMAGEM
        directories_selezionate = request.form.getlist('directories_selezionate')
        if not directories_selezionate:
            # Se o usuário não marcou nada, usar todas
            directories_selezionate = [d["path"] for d in directories_indicizzate]

        import time
        timestamp = int(time.time())
        temp_filename = f"temp_upload_{timestamp}.jpg"
        temp_filepath = os.path.join("/tmp", temp_filename)
        image_file.save(temp_filepath)

        try:
            embedding_img = extrai_features_imagem(temp_filepath)
            risultati, total_resultados = cerca_per_embedding(
                embedding_img,
                file_indice_faiss,
                file_percorsi,
                directories_selezionate=directories_selezionate,
                offset=0,
                limit=100
            )
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)

            if total_resultados == 0:
                flash("Nessuna immagine simile trovata.", 'info index')
                return redirect(url_for('index'))

            return render_template('risultati.html',
                                   immagini=risultati,
                                   total_resultados=total_resultados,
                                   per_page=100)
        except Exception as e:
            flash(f"Errore durante la ricerca per immagine: {e}", 'error index')
            logger.error("Errore durante la ricerca per immagine: %s", e)
            return redirect(url_for('index'))

    else:
        # BUSCA POR TEXTO
        testo_ricerca = request.form.get('input_testo', '')
        categoria = request.form.get('categoria', '')
        directories_selezionate = request.form.getlist('directories_selezionate')
        tags_selezionate = request.form.get('tags_selezionate', '')
        tags_list = tags_selezionate.split(',') if tags_selezionate else []

        if not directories_indicizzate:
            flash('Errore: Devi indicizzare prima le immagini.', 'error index')
            return redirect(url_for('index'))

        if not directories_selezionate:
            flash('Errore: Seleziona almeno una directory per la ricerca.', 'error index')
            return redirect(url_for('index'))

        testo_completo = testo_ricerca
        if tags_list:
            testo_completo += ' ' + ' '.join(tags_list)

        per_page = 100
        session['termo_busca'] = testo_completo
        session['categoria'] = categoria
        session['per_page'] = per_page
        session['directories_selezionate'] = directories_selezionate

        try:
            logger.debug("Iniciando busca (TEXTO): '%s', categoria='%s'", testo_completo, categoria)
            immagini, total_resultados = cerca_immagini(
                descrizione=testo_completo,
                categoria=categoria,
                file_indice_faiss=file_indice_faiss,
                file_percorsi=file_percorsi,
                directories_selezionate=directories_selezionate,
                offset=0,
                limit=per_page
            )
            session['total_resultados'] = total_resultados
            logger.debug("Busca retornou %d imagens, total: %d", len(immagini), total_resultados)

            return render_template('risultati.html',
                                   immagini=immagini,
                                   total_resultados=total_resultados,
                                   per_page=per_page)
        except Exception as e:
            flash(f"Errore durante la ricerca: {e}", 'error index')
            logger.error("Errore durante a pesquisa: %s", e)
            return redirect(url_for('index'))

@app.route('/carregar_mais_imagens', methods=['POST'])
@login_required
def carregar_mais_imagens():
    data = request.get_json()
    current_index = data.get('current_index', 0)
    per_page = data.get('per_page', 100)
    termo_busca = session.get('termo_busca', '')
    categoria = session.get('categoria', '')
    directories_selezionate = session.get('directories_selezionate', [])
    tags_selezionate = session.get('tags_selezionate', '')
    tags_list = tags_selezionate.split(',') if tags_selezionate else []
    testo_completo = termo_busca
    if tags_list:
        testo_completo += ' ' + ' '.join(tags_list)

    if not testo_completo and not categoria:
        return jsonify({'imagens': [], 'end': True})

    immagini, total_resultados = cerca_immagini(
        descrizione=testo_completo,
        categoria=categoria,
        file_indice_faiss=file_indice_faiss,
        file_percorsi=file_percorsi,
        directories_selezionate=directories_selezionate,
        offset=current_index,
        limit=per_page
    )
    end_of_results = (current_index + len(immagini)) >= total_resultados
    return jsonify({'imagens': immagini, 'end': end_of_results})

@app.route('/immagini/<path:percorso>')
@login_required
def immagini(percorso):
    percorso = '/' + percorso
    if not percorso:
        return "Immagine non trovata", 404
    if os.path.exists(percorso):
        return send_file(percorso)
    else:
        return "Immagine non trovata", 404

@app.route('/miniatura/<path:percorso>')
@login_required
def miniatura(percorso):
    percorso = '/' + percorso
    tamanho = (200, 200)
    if os.path.exists(percorso):
        with Image.open(percorso) as img:
            img.thumbnail(tamanho)
            img_io = BytesIO()
            img.save(img_io, 'JPEG')
            img_io.seek(0)
            return send_file(img_io, mimetype='image/jpeg')
    else:
        return "Immagine non trovata", 404

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

# ======================================================
#   DUPLICATAS (encontrar_duplicatas, eliminar_duplicatas)
# ======================================================
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
                        # Só deleta se pertence a alguma directory indicizzata
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
                    # Nenhuma imagem sobrou => Remove todos os arquivos
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

# ======================================================
#   PÁGINA DE CATEGORIE E TAG
# ======================================================
@app.route('/configura_categorie', methods=['GET', 'POST'])
@login_required
def configura_categorie():
    global categorie
    if request.method == 'POST':
        data = request.form.get('categorie_data')
        try:
            nuove_categorie = json.loads(data)
            with open(categorie_file, 'w') as f:
                json.dump(nuove_categorie, f, ensure_ascii=False, indent=4)
            categorie = carica_categorie()
            flash('Categorie e tag aggiornate con successo.', 'success')
            return redirect(url_for('configura_categorie'))
        except json.JSONDecodeError:
            flash('Errore: Il formato JSON non è valido.', 'error')
            return redirect(url_for('configura_categorie'))
        except Exception as ex:
            flash(f'Errore durante l\'aggiornamento: {ex}', 'error')
            return redirect(url_for('configura_categorie'))
    else:
        if not os.path.exists(categorie_file):
            with open(categorie_file, 'w') as f:
                json.dump({}, f, ensure_ascii=False, indent=4)
        with open(categorie_file, 'r') as f:
            try:
                categorie_data = json.dumps(categorie, ensure_ascii=False, indent=4)
            except:
                categorie_data = '{}'
        return render_template('configura_categorie.html', categorie_data=categorie_data)

# ======================================================
#   PROGRAMMAZIONE INDICIZZAZIONE (APSCHEDULER)
# ======================================================
day_map_ita = {
    "Lunedì": "mon",
    "Martedì": "tue",
    "Mercoledì": "wed",
    "Giovedì": "thu",
    "Venerdì": "fri",
    "Sabato": "sat",
    "Domenica": "sun"
}

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
    logger.debug("job_reindicizza_programmata() disparado pelo APScheduler!")
    with reindicizzazione_programmata_lock:
        reindicizzazione_programmata_progress['percentuale'] = 0
        reindicizzazione_programmata_progress['log'] = ["Reindicizzazione programmata avviata..."]
        reindicizzazione_programmata_progress['completed'] = False

    def run_scheduled():
        try:
            logger.debug("Iniciando indicizza_immagini (reindex programada), dirs= %s", directories_indicizzate)
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
    logger.debug("start_scheduler() chamado. Carregando pianificazione...")
    carica_pianificazione()
    days = schedule_data.get("days", [])
    hour_str = schedule_data.get("hour", "00:00")

    if not days or not hour_str:
        logger.debug("Nenhum dia/hora programado => Scheduler não criou job")
        return

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

    logger.debug("Programando reindex com day_of_week=%s, hour=%d, minute=%d", day_of_week, h, m)

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
    logger.debug("Programma salvata => Chamando start_scheduler() novamente para recriar job do APScheduler")
    start_scheduler()
    flash('Programmazione salvata con successo.', 'success')
    return redirect(url_for('programmazione_indicizzazione'))

# ==================== ROTTE PER IL CONVERSOR ==========================
def run_conversion(input_dir, output_dir):
    with conversion_lock:
        conversion_status['percentuale'] = 0
        conversion_status['log'] = []
        conversion_status['completed'] = False
        conversion_status['total_files'] = 0
        conversion_status['processed_files'] = 0

    cmd_find = [
        "find", input_dir,
        "-type", "d", "-name", ".*", "-prune", "-o",
        "-type", "f",
        "(",
            "-iname", "*.psd", "-o",
            "-iname", "*.psb", "-o",
            "-iname", "*.ifd", "-o",
            "-iname", "*.tif", "-o",
            "-iname", "*.tiff",
        ")",
        "-print"
    ]
    result = subprocess.run(cmd_find, capture_output=True, text=True)
    files_list = result.stdout.strip().split('\n')
    total_files = len([f for f in files_list if f.strip() != ''])
    with conversion_lock:
        conversion_status['total_files'] = total_files
        conversion_status['log'].append(f"Trovati {total_files} file da convertire.")

    script_path = os.path.join(os.path.dirname(__file__), "app", "convert.sh")
    env_vars = os.environ.copy()
    env_vars["INPUT_DIR"] = input_dir
    env_vars["OUTPUT_DIR"] = output_dir

    process = subprocess.Popen(
        ["bash", script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env_vars
    )
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            line = line.strip('\n')
            with conversion_lock:
                if line.startswith("Convertendo:"):
                    conversion_status['processed_files'] += 1
                    if total_files > 0:
                        p = int((conversion_status['processed_files'] / total_files)*100)
                    else:
                        p = 0
                    conversion_status['percentuale'] = p
                conversion_status['log'].append(line)
    process.wait()
    with conversion_lock:
        conversion_status['percentuale'] = 100
        conversion_status['completed'] = True
        conversion_status['log'].append("Conversão finalizzata.")

@app.route('/conversor')
@login_required
def conversor():
    return render_template('conversor.html')

@app.route('/inicia_conversao', methods=['POST'])
@login_required
def inicia_conversao():
    global conversion_thread
    input_dir = request.form.get('input_dir', '').strip()
    output_dir = request.form.get('output_dir', '').strip()
    if not input_dir:
        flash("Errore: cartella di input obbligatoria.", "error")
        return redirect(url_for('conversor'))
    if not output_dir:
        flash("Errore: cartella di output obbligatoria.", "error")
        return redirect(url_for('conversor'))
    if not os.path.isdir(input_dir):
        flash("Errore: La cartella di input non esiste o non è valida.", "error")
        return redirect(url_for('conversor'))
    if not os.path.isdir(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception as e:
            flash(f"Errore al creare la cartella di output: {e}", "error")
            return redirect(url_for('conversor'))
    if conversion_thread and conversion_thread.is_alive():
        flash("Conversione già in corso!", "info")
        return redirect(url_for('conversor'))

    def thread_func():
        run_conversion(input_dir, output_dir)

    conversion_thread = threading.Thread(target=thread_func)
    conversion_thread.start()
    flash("Conversione avviata. Guarda il log qui sotto.", "success")
    return redirect(url_for('conversor'))

# ==================== EXTRA: Informazioni di Immagini ====================
info_immagini_file = 'info_immagini.json'

def carica_info_immagini():
    if os.path.exists(info_immagini_file):
        try:
            with open(info_immagini_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def salva_info_immagini(data_dict):
    with open(info_immagini_file, 'w') as f:
        json.dump(data_dict, f, indent=4, ensure_ascii=False)

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

@app.route('/stato_conversao')
@login_required
def stato_conversao():
    with conversion_lock:
        return jsonify({
            'percentuale': conversion_status['percentuale'],
            'log': conversion_status['log'],
            'completed': conversion_status['completed'],
            'total_files': conversion_status['total_files'],
            'processed_files': conversion_status['processed_files']
        })

# ======================================================
#   PARAMETRI
# ======================================================
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
        'percorsi_immagini.pkl'
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

# ======================== NOVA ROTA: LOGS ========================
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

# ======================================================
#   PÁGINA DE GESTÃO DE USUÁRIOS
# ======================================================
@app.route('/gestione_utenti', methods=['GET', 'POST'])
@login_required
def gestione_utenti():
    """
    Permite visualizzare la lista di utenti e aggiungere nuovi.
    """
    if request.method == 'POST':
        # Adicionar um novo usuário
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

# ============ NOVO: Import da parte generativa ============
try:
    # Instale as libs: diffusers, transformers, accelerate, torch, torchvision, safetensors
    from diffusers import StableDiffusionPipeline
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Carregando modelo Stable Diffusion (v1-5) para geração de imagens...")
    model_name = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device=="cuda" else torch.float32
    )
    pipe.to(device)
    logger.info(f"Modelo carregado com sucesso! Usando device={device}.")
except Exception as e:
    logger.error(f"Falha ao carregar modelo de difusão: {e}")
    pipe = None

# ============ NOVA ROTA E PÁGINA DE GERAÇÃO DE IMAGENS ============
@app.route('/geracao', methods=['GET', 'POST'])
@login_required
def geracao():
    # Função para listar últimas 12 imagens geradas no "static/generated"
    def ultimas_imagens_geradas(n=12):
        gen_path = os.path.join("static", "generated")
        if not os.path.isdir(gen_path):
            return []
        # Pega todos os .png
        files = [f for f in os.listdir(gen_path) if f.lower().endswith('.png')]
        # Ordena por data de modificação desc
        files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(gen_path, x)), reverse=True)
        return files[:n]

    if request.method == 'POST':
        if not pipe:
            flash('Il modello di generazione non è caricato oppure si è verificato un errore.', 'error')
            return render_template('geracao.html', generated_image=None, ultimas_imagens=ultimas_imagens_geradas())

        prompt = request.form.get('prompt', '').strip()
        risoluzione = request.form.get('risoluzione', '512x512').strip()

        if not prompt:
            flash('Inserisci un prompt valido.', 'error')
            return render_template('geracao.html', generated_image=None, ultimas_imagens=ultimas_imagens_geradas())

        # Parse da resolucao
        # Exemplo "1920x1080" => h=1080, w=1920
        try:
            w_str, h_str = risoluzione.lower().split('x')
            width = int(w_str)
            height = int(h_str)
        except:
            # fallback
            width, height = 512, 512

        # Gera a imagem
        try:
            import torch
            if device == "cuda":
                with torch.autocast("cuda"):
                    result = pipe(prompt, height=height, width=width, num_inference_steps=30)
                    image = result.images[0]
            else:
                result = pipe(prompt, height=height, width=width, num_inference_steps=30)
                image = result.images[0]

            # Salva
            import time, random, string
            random_str = ''.join(random.choice(string.ascii_lowercase) for _ in range(6))
            filename = f"generated_{int(time.time())}_{random_str}.png"
            output_dir = os.path.join("static", "generated")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)
            image.save(output_path)

            flash('Immagine generata con successo!', 'success')
            return render_template(
                'geracao.html',
                generated_image=url_for('static', filename=f"generated/{filename}"),
                ultimas_imagens=ultimas_imagens_geradas()
            )
        except Exception as e:
            logger.error(f"Erro generando immagine: {e}")
            flash(f"Errore durante la generazione dell'immagine: {e}", 'error')
            return render_template('geracao.html', generated_image=None, ultimas_imagens=ultimas_imagens_geradas())

    else:
        # GET
        return render_template('geracao.html', generated_image=None, ultimas_imagens=ultimas_imagens_geradas())



# ============ NOVA SEÇÃO: CONFIGURAÇÃO DA GENERAÇÃO (LoRA, etc.) ============

@app.route("/stato_lora")
@login_required
def stato_lora():
    with lora_lock:
        return jsonify({
            "running": lora_progress["running"],
            "percent": lora_progress["percent"],
            "log": lora_progress["log"],
            "completed": lora_progress["completed"]
        })



@app.route('/configurazione_generativa', methods=['GET', 'POST'])
@login_required
def configurazione_generativa():
    """
    Página onde o usuário escolhe quais diretórios indexados vão participar da
    geração. Ex.: geramos e salvamos esse status em 'generative_dirs' no config.
    Também há um botão para 'indexazione incrementale' (exemplo).
    """
    if request.method == 'POST':
        # Recebe checkboxes
        form_data = request.form
        # Example: "enabled_<path>" = 'on'
        generative_data = {}
        for d in directories_indicizzate:
            path_ = d["path"]
            checkbox_name = f"enable_{path_}"
            if checkbox_name in form_data:
                generative_data[path_] = True
            else:
                generative_data[path_] = False

        salva_configurazione_generativa(generative_data)
        flash("Configurazione generativa salvata con successo!", "success")
        return redirect(url_for('configurazione_generativa'))

    else:
        # GET => exibe a página
        generative_data = carica_configurazione_generativa()
        return render_template(
            "configurazione_generativa.html",
            directories_indicizzate=directories_indicizzate,
            generative_data=generative_data
        )




# Exemplo de rota de "indexação incremental" para geração
@app.route('/indicizza_generativa', methods=['POST'])
@login_required
def indicizza_generativa():
    """
    Faz uma pseudo "indexação incremental" focada para geração.
    Aqui você poderia disparar um script de treino LoRA, etc.
    Exemplo simples: logar e dar flash.
    """
    generative_data = carica_configurazione_generativa()
    # Filtrar só as que estão 'true'
    dirs_ativos = [k for k,v in generative_data.items() if v]
    # Exemplo: rodar alguma rotina de "treino LoRA" nessas dirs_ativos...
    logger.info(f"[GEN] Indexacao/treino gerativo com as dirs = {dirs_ativos}")
    flash(f"Indexazione incrementale (generativa) avviata per {len(dirs_ativos)} directory.", "info")

    return redirect(url_for('configurazione_generativa'))

@app.route('/treino_lora', methods=['POST'])
@login_required
def treino_lora():
    """
    Roda train_lora.py em segundo plano (thread), exibe barra de progresso na página.
    """
    global lora_thread, lora_progress

    # Lê pastas ativas
    generative_data = carica_configurazione_generativa()
    selected_dirs = [d for d, enabled in generative_data.items() if enabled]
    if not selected_dirs:
        flash("Nessuna directory selezionata per il training LoRA!", "error")
        return redirect(url_for('configurazione_generativa'))

    # Cria tmp folder
    tmp_folder = f"/tmp/train_lora_dataset_{int(time.time())}"
    os.makedirs(tmp_folder, exist_ok=True)
    logger.info(f"Treino LoRA - Pasta temporária: {tmp_folder}")

    # Copia imagens
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    total_imagens = 0
    for dir_ in selected_dirs:
        for root, dirs, files in os.walk(dir_):
            for fname in files:
                if fname.lower().endswith(exts):
                    src_path = os.path.join(root, fname)
                    new_name = os.path.basename(dir_) + "_" + fname
                    dest_path = os.path.join(tmp_folder, new_name)
                    try:
                        shutil.copy2(src_path, dest_path)
                        total_imagens += 1
                    except Exception as e:
                        logger.error(f"Erro copiando {src_path} => {dest_path}: {e}")

    if total_imagens == 0:
        flash("Não há imagens nas diretorias selecionadas para treino!", "error")
        return redirect(url_for('configurazione_generativa'))

    # Monta cmd
    random_tag = ''.join(random.choice(string.ascii_lowercase) for _ in range(6))
    output_lora_dir = f"./lora_output_{random_tag}"
    instance_prompt = "fabricStyle"
    cmd = [
        "accelerate", "launch", "train_lora.py",
        "--pretrained_model_name=runwayml/stable-diffusion-v1-5",
        f"--train_data_dir={tmp_folder}",
        f"--output_dir={output_lora_dir}",
        f"--instance_prompt={instance_prompt}",
        "--resolution=512",
        "--train_batch_size=1",
        "--gradient_accumulation_steps=1",
        "--learning_rate=1e-4",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        "--max_train_steps=1000",
    ]

    logger.info(f"[TREINO_LORA] Iniciando thread com cmd: {' '.join(cmd)}")

    # Reseta status
    with lora_lock:
        lora_progress["running"] = True
        lora_progress["percent"] = 0
        lora_progress["log"] = []
        lora_progress["completed"] = False

    def run_lora_training():
        """
        Função que efetivamente chama o train_lora.py com Popen e
        atualiza lora_progress em tempo real.
        """
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            # Ler stdout linha a linha
            total_steps = 1000  # se quisermos deduzir % real, ou parse do log
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    line = line.rstrip("\n")
                    with lora_lock:
                        lora_progress["log"].append(line)

                        # Exemplo: se encontrar "Step X/1000, loss=..." no log
                        # podemos extrair X e calcular %.
                        if "Step " in line and "loss=" in line:
                            # Tenta parse do tipo "Step 50/1000, loss=0.1234"
                            # isso depende do print do script train_lora.py
                            try:
                                # localiza "Step 50/1000"
                                part = line.split("Step ")[1].split(",")[0].strip()
                                # "50/1000"
                                current, total = part.split("/")
                                current_step = int(current)
                                max_steps = int(total)
                                perc = int((current_step / max_steps)*100)
                                lora_progress["percent"] = perc
                            except:
                                pass

            ret = process.wait()
            with lora_lock:
                lora_progress["completed"] = True
                lora_progress["running"] = False
                if ret == 0:
                    lora_progress["log"].append(f"Treino LoRA finalizado com sucesso! Output em: {output_lora_dir}")
                else:
                    lora_progress["log"].append(f"Treino LoRA retornou erro (code={ret}).")
        except Exception as e:
            with lora_lock:
                lora_progress["completed"] = True
                lora_progress["running"] = False
                lora_progress["log"].append(f"Exceção no treino LoRA: {e}")

    # Dispara thread
    global lora_thread
    lora_thread = threading.Thread(target=run_lora_training)
    lora_thread.start()

    flash("Treino LoRA iniciado em segundo plano!", "info")
    # Agora redireciona à mesma página, que fará AJAX para /stato_lora
    return redirect(url_for('configurazione_generativa'))


# ======================== Inicialização (Waitress) ========================
if __name__ == '__main__':
    # Garante que o usuário padrão exista (chickellero)
    ensure_default_user()

    from waitress import serve
    logger.debug("[DEBUG] __main__ -> Chamando start_scheduler() para APScheduler.")
    start_scheduler()
    logger.debug("[DEBUG] Subindo waitress server na porta 5000...")
    serve(app, host='0.0.0.0', port=5000, threads=8)
