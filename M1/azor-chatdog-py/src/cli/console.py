"""
Console output utilities for the chatbot.
Centralizes colorama usage for consistent terminal output.
"""
import sys
import shutil
import textwrap
from colorama import init, Fore, Style
from files.config import LOG_DIR

init(autoreset=True)


def _wrap(message: str) -> str:
    try:
        width = shutil.get_terminal_size((120, 20)).columns
    except Exception:
        width = 120
    width = max(50, min(width, 160))
    wrapped_lines = []
    for line in message.splitlines():
        stripped = line.strip()
        if stripped.startswith('```') or stripped.startswith('{') or stripped.startswith('    '):
            # Leave code fences / JSON / indented blocks untouched
            wrapped_lines.append(line)
            continue
        if len(line) <= width:
            wrapped_lines.append(line)
        else:
            wrapped_lines.extend(textwrap.wrap(line, width=width))
    return '\n'.join(wrapped_lines)

def print_error(message: str):
    print(Fore.RED + _wrap(message) + Style.RESET_ALL)


def print_assistant(message: str):
    print(Fore.CYAN + _wrap(message) + Style.RESET_ALL)


def print_user(message: str):
    print(Fore.BLUE + _wrap(message) + Style.RESET_ALL)


def print_info(message: str):
    print(_wrap(message))

def print_help(message: str):
    print(Fore.YELLOW + _wrap(message) + Style.RESET_ALL)


def display_help(session_id: str):
    """Displays a short help message."""
    print_info(f"Aktualna sesja (ID): {session_id}")
    # Try read title quickly
    try:
        from files import session_files
        import os, json
        log_filename = os.path.join(session_files.LOG_DIR, f"{session_id}-log.json")
        if os.path.exists(log_filename):
            with open(log_filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                title = data.get('title')
                if title:
                    print_info(f"Tytuł sesji: {title}")
    except Exception:
        pass
    print_info(f"Pliki sesji są zapisywane na bieżąco w: {LOG_DIR}")
    print_help("Dostępne komendy (slash commands):")
    print_help("  /switch <ID>          - Przełącza na istniejącą sesję.")
    print_help("  /assistant <NAZWA>    - Zmienia aktywnego asystenta.")
    print_help("  /AZOR /PERFEKCJONISTA /BIZNESMEN /OPTIMISTA (/OPTYMISTA) - Bezpośrednie przełączenie.")
    print_help("  /help                 - Wyświetla tę pomoc.")
    print_help("  /exit, /quit          - Zakończenie czatu.")
    print_help("\n  /session list         - Lista dostępnych sesji.")
    print_help("  /session display      - Pełna historia bieżącej sesji.")
    print_help("  /session pop          - Usuwa ostatnią parę (TY + asystent).")
    print_help("  /session clear        - Czyści historię bieżącej sesji.")
    print_help("  /session new          - Nowa sesja.")
    print_help("  /session remove       - Usuwa plik bieżącej sesji i zaczyna nową.")
    print_help("  /pdf                  - Eksport do PDF.")
    print_help("  /audio                - Synteza mowy ostatniej odpowiedzi.")
    print_help("  /title <NOWY_TYTUŁ>    - Ręczna zmiana tytułu bieżącej sesji.")


def display_final_instructions(session_id: str):
    """Displays instructions for continuing the session."""
    print_info("\n--- Instrukcja Kontynuacji Sesji ---")
    print_info(f"Aby kontynuować tę sesję (ID: {session_id}) później, użyj komendy:")
    print(Fore.WHITE + Style.BRIGHT + f"\n    python {sys.argv[0]} --session-id={session_id}\n" + Style.RESET_ALL)
    print("--------------------------------------\n")

