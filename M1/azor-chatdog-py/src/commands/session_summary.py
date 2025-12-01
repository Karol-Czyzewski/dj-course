from typing import List, Dict
from cli import console
from files import session_files
import os, json

def display_history_summary(history: List[Dict], assistant_name: str, session_id: str | None = None):
    """
    Wyświetla podsumowanie historii: liczbę pominiętych i ostatnie 2 wiadomości.
    
    Args:
        history: Lista słowników w formacie {"role": "user|model", "parts": [{"text": "..."}]}
        assistant_name: Nazwa asystenta do wyświetlenia
    """
    total_count = len(history)
    
    if total_count == 0:
        return

    # Wyświetlenie podsumowania
    if total_count > 2:
        console.print_info(f"\n--- Wątek sesji wznowiony ---")
        omitted_count = total_count - 2
        console.print_info(f"(Pominięto {omitted_count} wcześniejszych wiadomości)")
    else:
        console.print_info(f"\n--- Wątek sesji ---")

    # Optional: read title for display
    title = None
    if session_id:
        try:
            log_filename = os.path.join(session_files.LOG_DIR, f"{session_id}-log.json")
            if os.path.exists(log_filename):
                with open(log_filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    title = data.get('title')
        except Exception:
            pass

    # Display last 2 messages
    last_two = history[-2:]
    
    for content in last_two:
        # Handle universal dictionary format
        role = content.get('role', '')
        display_role = "TY" if role == "user" else assistant_name
        
        # Extract text from parts
        text = ""
        if 'parts' in content and content['parts']:
            text = content['parts'][0].get('text', '')
        
        if role == "user":
            console.print_user(f"  {display_role}: {text}")
            if title:
                console.print_info(f"  📝 Tytuł wątku: {title}")
        elif role == "model":
            console.print_assistant(f"  {display_role}: {text}")
            
    console.print_info(f"----------------------------")

