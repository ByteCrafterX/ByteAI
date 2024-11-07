from flask import Flask, render_template, request, send_from_directory, redirect, url_for, flash, jsonify, abort, session
import os
import threading
import json
import subprocess

from modello import cerca_immagini, encontrar_duplicatas
from indicizza import indicizza_immagini
from progress import indicizzazione_progress, progress_lock

app = Flask(__name__)
app.secret_key = 'chiave_segreta_per_flash_message'

# Caminhos dos arquivos de índice
file_indice_faiss = 'indice_faiss.index'
file_percorsi = 'percorsi_immagini.pkl'
embeddings_file = 'embeddings_immagini.npy'  # Arquivo dos embeddings das imagens
config_file = 'config.json'  # Arquivo para salvar o diretório das imagens

# Variáveis para rastrear threads e progresso
indicizzazione_thread = None

# Variáveis globais para o progresso da eliminação
eliminazione_progress = {
    'percentuale': 0,
    'log': []
}
eliminazione_lock = threading.Lock()

# Funções para carregar e salvar a configuração
def carica_configurazione():
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
            return config.get('directory_immagini', '')
    return ''

def salva_configurazione(directory_immagini):
    with open(config_file, 'w') as f:
        json.dump({'directory_immagini': directory_immagini}, f)

# Inicializa o diretório base das imagens
directory_immagini = carica_configurazione()

# Página principal
@app.route('/')
def index():
    return render_template('index.html')

# Rota para a página "About"
@app.route('/about')
def about():
    return render_template('about.html')

# Rota para a página de configurações
@app.route('/configurazioni')
def configurazioni():
    return render_template('config.html')

# Rota para a página "Utilità"
@app.route('/utilidades')
def utilidades():
    return render_template('utilidades.html')

# Rota para iniciar a indexação
@app.route('/indicizza', methods=['POST'])
def indicizza():
    global indicizzazione_thread, directory_immagini
    directory_input = request.form.get('directory_indicizza')
    if not directory_input:
        flash('Errore: Il campo della directory è obbligatorio per l\'indicizzazione.', 'error')
        return redirect(url_for('configurazioni'))

    if not os.path.isdir(directory_input):
        flash('Errore: La directory fornita non è valida.', 'error')
        return redirect(url_for('configurazioni'))

    if indicizzazione_thread and indicizzazione_thread.is_alive():
        flash('Indicizzazione già in corso.', 'error')
        return redirect(url_for('configurazioni'))

    # Define o diretório base das imagens
    directory_immagini = directory_input

    # Salva o diretório no arquivo de configuração
    salva_configurazione(directory_immagini)

    # Reseta o progresso anterior
    with progress_lock:
        indicizzazione_progress['percentuale'] = 0
        indicizzazione_progress['log'] = []

    # Executa a indexação em uma thread separada
    def run_indicizzazione():
        try:
            indicizza_immagini(directory_immagini, file_indice_faiss, file_percorsi)
            with progress_lock:
                indicizzazione_progress['percentuale'] = 100
        except Exception as e:
            with progress_lock:
                indicizzazione_progress['log'].append(f"Errore durante l'indicizzazione: {e}")
            print(f"Errore durante l'indicizzazione: {e}")

    indicizzazione_thread = threading.Thread(target=run_indicizzazione)
    indicizzazione_thread.start()

    flash('Indicizzazione avviata.', 'success')
    return redirect(url_for('configurazioni'))

# Rota para obter o estado da indexação
@app.route('/stato_indicizzazione')
def stato_indicizzazione():
    with progress_lock:
        progress = indicizzazione_progress.copy()
    return jsonify(progress)

# Rota para a pesquisa
@app.route('/ricerca', methods=['POST'])
def ricerca():
    global directory_immagini
    testo_ricerca = request.form.get('input_testo')
    categoria = request.form.get('categoria')



    if not directory_immagini:
        flash('Errore: Devi indicizzare prima le immagini.', 'error')
        print("Errore: Directory immagini non indicizzata.")
        return redirect(url_for('index'))

    print(f"Testo di ricerca: {testo_ricerca}")
    print(f"Categoria selezionata: {categoria}")

    # Número de resultados por página
    per_page = 100  # Carrega 100 imagens inicialmente

    # Armazena os parâmetros de pesquisa na sessão
    session['termo_busca'] = testo_ricerca
    session['categoria'] = categoria
    session['per_page'] = per_page

    # Chama a função para buscar as imagens
    try:
        immagini, total_resultados = cerca_immagini(
            testo_ricerca, categoria, file_indice_faiss, file_percorsi,
            offset=0, limit=per_page
        )

        print(f"Total de risultati trovati: {total_resultados}")

        session['total_resultados'] = total_resultados

        # Ajusta os caminhos para serem relativos ao diretório base das imagens
        for img in immagini:
            img['percorso_relativo'] = os.path.relpath(img['percorso'], directory_immagini)

        return render_template('risultati.html', immagini=immagini, total_resultados=total_resultados)
    except Exception as e:
        flash(f"Errore durante la ricerca: {e}", 'error')
        print(f"Errore durante la ricerca: {e}")
        return redirect(url_for('index'))

# Rota para carregar mais imagens via AJAX
@app.route('/carregar_mais_imagens', methods=['POST'])
def carregar_mais_imagens():
    data = request.get_json()
    current_index = data.get('current_index', 0)
    per_page = data.get('per_page', 100)  # Carrega 100 imagens por vez

    termo_busca = session.get('termo_busca')
    categoria = session.get('categoria')

    print(f"Carregar mais imagens - Current index: {current_index}, Per page: {per_page}")

    if not termo_busca:
        print("Termo di ricerca non trovato nella sessione.")
        return jsonify({'imagens': [], 'end': True})

    # Chama a função para obter o próximo lote de imagens
    immagini, total_resultados = cerca_immagini(
        termo_busca, categoria, file_indice_faiss, file_percorsi,
        offset=current_index, limit=per_page
    )

    print(f"Numero di immagini restituite: {len(immagini)}")
    print(f"Total de risultati disponibili: {total_resultados}")

    # Ajusta os caminhos para serem relativos ao diretório base das imagens
    for img in immagini:
        img['percorso_relativo'] = os.path.relpath(img['percorso'], directory_immagini)

    end_of_results = current_index + len(immagini) >= total_resultados

    return jsonify({'imagens': immagini, 'end': end_of_results})

# Rota para servir as imagens
@app.route('/immagini/<path:percorso>')
def immagini(percorso):
    global directory_immagini
    if not directory_immagini:
        abort(404)
    percorso_completo = os.path.join(directory_immagini, percorso)

    if os.path.exists(percorso_completo):
        return send_from_directory(directory_immagini, percorso)
    else:
        return "Immagine non trovata", 404

# Rota para obter os metadados diretamente da imagem
@app.route('/get_metadata', methods=['POST'])
def get_metadata():
    data = request.get_json()
    image_path = data.get('image_path')
    if not image_path:
        return jsonify({'error': 'Percorso immagine non fornito'}), 400

    # Constrói o caminho completo da imagem
    full_image_path = os.path.join(directory_immagini, image_path)
    if not os.path.exists(full_image_path):
        return jsonify({'error': 'Immagine non trovata'}), 404

    try:
        # Usa o ExifTool para extrair os metadados XMP
        result = subprocess.run(
            ['exiftool', '-j', '-XMP:All', full_image_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:
            metadati = {'Errore': result.stderr}
        else:
            metadata_json = json.loads(result.stdout)
            if metadata_json:
                metadati = metadata_json[0]
                # Filtra apenas as chaves desejadas
                metadati_filtrati = {
                    'dc:subject': metadati.get('Subject', ''),
                    'dc:title': metadati.get('Title', ''),
                    'xmp:rating': metadati.get('Rating', '')
                }
                metadati = metadati_filtrati
            else:
                metadati = {'Metadati': 'Non disponibili.'}

    except Exception as e:
        metadati = {'Errore': str(e)}

    return jsonify({'metadati': metadati})

# Rota para encontrar duplicatas
@app.route('/encontrar_duplicatas', methods=['GET', 'POST'])
def encontrar_duplicatas_route():
    global directory_immagini
    if not directory_immagini:
        flash('Errore: Devi indicizzare prima le immagini.', 'error')
        return redirect(url_for('utilidades'))

    if request.method == 'POST':
        # Obtém o valor de similaridade inserido pelo usuário
        similaridade_input = request.form.get('similaridade')
        try:
            # Substitui vírgula por ponto para suportar separador decimal
            similaridade_input = similaridade_input.replace(',', '.')
            similaridade_valor = float(similaridade_input)
            if not 0 <= similaridade_valor <= 1:
                raise ValueError("Il valore deve essere compreso tra 0 e 1.")
        except ValueError as ve:
            flash(f'Errore: {ve}', 'error')
            return redirect(url_for('encontrar_duplicatas_route'))
        except Exception:
            flash('Errore: Inserisci un valore di similarità valido (usa il punto o la virgola come separatore decimale).', 'error')
            return redirect(url_for('encontrar_duplicatas_route'))
    else:
        # Valor padrão
        similaridade_valor = 0.95

    # Armazena o valor de similaridade na sessão
    session['similaridade_valor'] = similaridade_valor

    # Chama a função para encontrar duplicatas
    try:
        grupos_duplicatas = encontrar_duplicatas(embeddings_file, file_percorsi, threshold_duplicates=similaridade_valor)
        total_duplicatas = sum(len(grupo) - 1 for grupo in grupos_duplicatas)  # Exclui a imagem original

        # Ajusta os caminhos para serem relativos ao diretório base das imagens
        for grupo in grupos_duplicatas:
            for img in grupo:
                img['percorso_relativo'] = os.path.relpath(img['percorso'], directory_immagini)

        if total_duplicatas == 0:
            flash('Nessuna immagine duplicata trovata.', 'info')
            return render_template('duplicatas.html', grupos=[], total_duplicatas=0, similaridade=similaridade_valor)

        return render_template('duplicatas.html', grupos=grupos_duplicatas, total_duplicatas=total_duplicatas, similaridade=similaridade_valor)
    except Exception as e:
        flash(f"Errore durante la ricerca di duplicati: {e}", 'error')
        print(f"Errore durante la ricerca di duplicati: {e}")
        return redirect(url_for('utilidades'))

# Rota para eliminar duplicatas
@app.route('/eliminar_duplicatas')
def eliminar_duplicatas_route():
    global directory_immagini, eliminazione_progress
    if not directory_immagini:
        flash('Errore: Devi indicizzare prima le immagini.', 'error')
        return redirect(url_for('utilidades'))

    # Obter o valor de similaridade dos parâmetros da URL
    similaridade_input = request.args.get('similarity', '0.95')
    try:
        # Substituir vírgula por ponto para suportar separador decimal
        similaridade_input = similaridade_input.replace(',', '.')
        similaridade_valor = float(similaridade_input)
        if not 0 <= similaridade_valor <= 1:
            raise ValueError("Il valore deve essere compreso tra 0 e 1.")
    except ValueError as ve:
        flash(f'Errore: {ve}', 'error')
        return redirect(url_for('encontrar_duplicatas_route'))
    except Exception:
        flash('Errore: Inserisci un valore di similarità valido (usa il punto o la virgola come separatore decimale).', 'error')
        return redirect(url_for('encontrar_duplicatas_route'))

    def run_eliminazione():
        try:
            # Inicializa o progresso
            with eliminazione_lock:
                eliminazione_progress['percentuale'] = 0
                eliminazione_progress['log'] = []

            # Chama a função para encontrar duplicatas com o valor de similaridade fornecido
            grupos_duplicatas = encontrar_duplicatas(
                embeddings_file,
                file_percorsi,
                threshold_duplicates=similaridade_valor
            )

            # Coleta os caminhos das imagens duplicadas (excluindo as imagens originais)
            percorsi_da_eliminare = []
            for gruppo in grupos_duplicatas:
                # Ordena o grupo por data de modificação para manter a versão mais recente
                gruppo_ordenato = sorted(gruppo, key=lambda x: x['data_modifica'], reverse=True)
                # Mantém a primeira imagem (mais recente) e marca as demais para exclusão
                immagini_da_eliminare = gruppo_ordenato[1:]
                for img in immagini_da_eliminare:
                    percorsi_da_eliminare.append(img['percorso'])

            total_para_eliminar = len(percorsi_da_eliminare)
            contagem_eliminados = 0

            # Remove as imagens duplicadas com atualização de progresso
            for idx, percorso in enumerate(percorsi_da_eliminare):
                if os.path.exists(percorso):
                    os.remove(percorso)
                    contagem_eliminados += 1

                # Atualiza o progresso
                with eliminazione_lock:
                    percentuale = int((idx + 1) / total_para_eliminar * 100)
                    eliminazione_progress['percentuale'] = percentuale
                    eliminazione_progress['log'].append(f"Eliminata {idx + 1} su {total_para_eliminar} immagini.")

            # Reindexa as imagens após a exclusão
            indicizza_immagini(directory_immagini, file_indice_faiss, file_percorsi)

            with eliminazione_lock:
                eliminazione_progress['log'].append("Eliminazione completata con successo.")
                eliminazione_progress['percentuale'] = 100

            flash(f'Sono state eliminate {contagem_eliminados} immagini duplicate.', 'success')

        except Exception as e:
            with eliminazione_lock:
                eliminazione_progress['log'].append(f"Errore durante l'eliminazione: {e}")
            flash(f"Errore durante l'eliminazione delle duplicati: {e}", 'error')
            print(f"Errore durante l'eliminazione delle duplicati: {e}")

    # Inicia a thread de eliminação
    eliminazione_thread = threading.Thread(target=run_eliminazione)
    eliminazione_thread.start()

    return redirect(url_for('stato_eliminazione_page'))

# Rota para obter o estado da eliminação
@app.route('/stato_eliminazione')
def stato_eliminazione():
    with eliminazione_lock:
        progress = eliminazione_progress.copy()
    return jsonify(progress)

# Rota para a página di stato dell'eliminazione
@app.route('/stato_eliminazione_page')
def stato_eliminazione_page():
    return render_template('stato_eliminazione.html')

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=5000, threads=8)
