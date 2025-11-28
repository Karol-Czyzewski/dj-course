from session import get_session_manager
from cli import console
from commands.session_list import list_sessions_command
from commands.session_display import display_full_session
from commands.session_to_pdf import export_session_to_pdf
from commands.session_remove import remove_session_command
from commands.audio import generate_audio_from_last_assistant
from assistant import available_assistants

VALID_SLASH_COMMANDS = ['/exit', '/quit', '/switch', '/help', '/session', '/pdf', '/audio', '/assistant']
# Dynamic assistant shortcut commands (e.g. /AZOR)
ASSISTANT_SHORTCUTS = [f"/{name}" for name in available_assistants()] + ["/OPTYMISTA"]  # include alias

def handle_command(user_input: str) -> bool:
    """
    Handles slash commands. Returns True if the program should exit.
    """
    parts = user_input.split()
    command = parts[0].lower()

    manager = get_session_manager()

    # Direct assistant shortcut (e.g. /AZOR)
    if command.upper() in [c.upper() for c in ASSISTANT_SHORTCUTS]:
        target_name = command[1:]  # strip leading '/'
        current = manager.get_current_session()
        old_name = current.assistant_name
        success, error = manager.switch_assistant_in_current_session(target_name)
        if not success:
            console.print_error(error or "Nie udało się przełączyć asystenta.")
        else:
            console.print_info(f"Przełączono asystenta: {old_name} → {current.assistant_name}")
        return False

    # Check if the main command is valid (non-assistant)
    if command not in VALID_SLASH_COMMANDS:
        console.print_error(f"Błąd: Nieznana komenda: {command}. Użyj /help.")
        current = manager.get_current_session()
        console.display_help(current.session_id)
        return False

    # Help command
    elif command == '/help':
        current = manager.get_current_session()
        console.display_help(current.session_id)

    # Exit commands
    if command in ['/exit', '/quit']:
        console.print_info("\nZakończenie czatu. Uruchamianie procedury finalnego zapisu...")
        return True

    # Switch command
    elif command == '/switch':
        if len(parts) == 2:
            new_id = parts[1]
            current = manager.get_current_session()
            if new_id == current.session_id:
                console.print_info("Jesteś już w tej sesji.")
            else:
                new_session, save_attempted, previous_session_id, load_successful, load_error, has_history = manager.switch_to_session(new_id)

                # Handle console output for save attempt
                if save_attempted:
                    console.print_info(f"\nZapisuję bieżącą sesję: {previous_session_id}...")

                # Handle load result
                if not load_successful:
                    console.print_error(f"Nie można wczytać sesji o ID: {new_id}. {load_error}")
                else:
                    # Successfully switched
                    console.print_info(f"\n--- Przełączono na sesję: {new_session.session_id} ---")
                    console.display_help(new_session.session_id)

                    # Display history summary if session has content
                    if has_history:
                        from commands.session_summary import display_history_summary
                        display_history_summary(new_session.get_history(), new_session.assistant_name)
        else:
            console.print_error("Błąd: Użycie: /switch <SESSION-ID>")

    # Session subcommands
    elif command == '/session':
        if len(parts) < 2:
            console.print_error("Błąd: Komenda /session wymaga podkomendy (list, display, pop, clear, new).")
        else:
            handle_session_subcommand(parts[1].lower(), manager)

    elif command == '/pdf':
        current = manager.get_current_session()
        export_session_to_pdf(current.get_history(), current.session_id, current.assistant_name)

    elif command == '/audio':
        current = manager.get_current_session()
        generate_audio_from_last_assistant(current.get_history(), current.session_id, current.assistant_name)

    elif command == '/assistant':
        # Switch current assistant within this session
        if len(parts) == 2:
            target = parts[1]
            current = manager.get_current_session()
            old_name = current.assistant_name
            success, error = manager.switch_assistant_in_current_session(target)
            if not success:
                console.print_error(error or "Nie udało się przełączyć asystenta.")
            else:
                console.print_info(f"Przełączono asystenta: {old_name} → {current.assistant_name}")
                console.print_info(f"Dostępni asystenci: {', '.join(available_assistants())}")
        else:
            console.print_error("Błąd: Użycie: /assistant <NAZWA>. Dostępni: " + ", ".join(available_assistants()))

    return False


def handle_session_subcommand(subcommand: str, manager):
    """Handles /session subcommands."""
    current = manager.get_current_session()

    if subcommand == 'list':
        list_sessions_command()

    elif subcommand == 'display':
        display_full_session(current.get_history(), current.session_id, current.assistant_name)

    elif subcommand == 'pop':
        success = current.pop_last_exchange()
        if success:
            from commands.session_summary import display_history_summary
            console.print_info(f"Usunięto ostatnią parę wpisów (TY i {current.assistant_name}).")
            display_history_summary(current.get_history(), current.assistant_name)
        else:
            console.print_error("Błąd: Historia jest pusta lub niekompletna (wymaga co najmniej jednej pary).")

    elif subcommand == 'clear':
        current.clear_history()
        console.print_info("Historia bieżącej sesji została wyczyszczona.")

    elif subcommand == 'new':
        new_session, save_attempted, previous_session_id, save_error = manager.create_new_session(save_current=True)

        # Handle console output for save attempt
        if save_attempted:
            console.print_info(f"\nZapisuję bieżącą sesję: {previous_session_id} przed rozpoczęciem nowej...")
            if save_error:
                console.print_error(f"Błąd podczas zapisu: {save_error}")

        # Display new session info
        console.print_info(f"\n--- Rozpoczęto nową sesję: {new_session.session_id} ---")
        console.display_help(new_session.session_id)

    elif subcommand == 'remove':
        remove_session_command(manager)

    else:
        console.print_error(f"Błąd: Nieznana podkomenda dla /session: {subcommand}. Użyj /help.")
