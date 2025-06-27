from functools import wraps
from flask import session, redirect, url_for


def login_required(f):
    """Protegge una view: se l'utente non è loggato, lo rimanda alla home."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("index"))
        return f(*args, **kwargs)

    return decorated_function

