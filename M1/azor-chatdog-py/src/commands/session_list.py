from files import session_files
from cli import console

def list_sessions_command():
    """Displays a formatted list of available sessions."""
    sessions = session_files.list_sessions()
    if sessions:
        console.print_help("\n--- Dostępne zapisane sesje (ID) ---")
        for session in sessions:
            if session.get('error'):
                console.print_error(f"- ID: {session['id']} ({session['error']})")
            else:
                # Wczytaj tytuł bez pełnego parsowania historii
                log_path = session_files.os.path.join(session_files.LOG_DIR, f"{session['id']}-log.json")
                title = None
                try:
                    with open(log_path, 'r', encoding='utf-8') as f:
                        data = session_files.json.load(f)
                        title = data.get('title')
                except Exception:
                    pass
                title_display = f" | Tytuł: {title}" if title else ""
                console.print_help(f"- ID: {session['id']}{title_display} (Wiadomości: {session['messages_count']}, Ost. aktywność: {session['last_activity']})")
        console.print_help("------------------------------------")
    else:
        console.print_help("\nBrak zapisanych sesji.")
