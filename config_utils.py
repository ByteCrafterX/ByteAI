# config_utils.py
import json
import os

CONFIG_FILE = "config.json"


def _load() -> dict:
    if not os.path.exists(CONFIG_FILE):
        return {}
    try:
        with open(CONFIG_FILE, "r") as fp:
            return json.load(fp)
    except json.JSONDecodeError:
        return {}


def _save(data: dict):
    with open(CONFIG_FILE, "w") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=4)


# ---------------- API generiche ---------------- #
def cfg_get(key: str, default=None):
    return _load().get(key, default)


def cfg_set(key: str, value):
    data = _load()
    data[key] = value
    _save(data)


# -------------- scorciatoie progetto ----------- #
def get_generative_dirs() -> dict:
    """Restituisce {path: bool} salvato in config.json."""
    return cfg_get("generative_dirs", {})


def set_generative_dirs(d: dict):
    cfg_set("generative_dirs", d)


def get_directories_indicizzate():
    """Lista delle directory già indicizzate (come nel vecchio codice)."""
    return cfg_get("directories_indicizzate", [])