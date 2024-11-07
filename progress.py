import threading

indicizzazione_progress = {
    'percentuale': 0,
    'log': []
}

progress_lock = threading.Lock()
