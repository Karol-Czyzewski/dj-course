from typing import List, Dict
import os
import subprocess
from datetime import datetime
from cli import console
from files.config import OUTPUT_DIR

"""Generowanie audio z ostatniej odpowiedzi asystenta.

Obsługa dwóch trybów:
1. Lokalny import biblioteki `coqui-tts` (jeśli zgodna wersja Pythona)
2. Fallback: wywołanie osobnego interpretera Python 3.12 (XTTS_PYTHON) uruchamiającego skrypt
   `scripts/xtts_generate.py` w celu wygenerowania pliku.

Zmiennie środowiskowe:
  XTTS_LANGUAGE     -> kod języka (domyślnie 'pl')
  XTTS_SPEAKER_WAV  -> ścieżka do wzorcowego głosu (opcjonalnie)
  XTTS_MODEL_NAME   -> nazwa modelu XTTS (domyślnie xtts_v2)
  XTTS_PYTHON       -> pełna ścieżka do interpretera Python 3.12.x z zainstalowanym coqui-tts
"""

def _extract_text_from_entry(entry: Dict) -> str:
    if not entry:
        return ""
    parts = entry.get('parts', [])
    if parts and isinstance(parts, list):
        first = parts[0]
        if isinstance(first, dict):
            return first.get('text', '') or ''
    return ''

def _find_last_assistant_reply(history: List[Dict]) -> str:
    for entry in reversed(history):
        if entry.get('role') == 'model':
            return _extract_text_from_entry(entry)
    return ''

def _build_output_path(session_id: str) -> str:
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    file_name = f"{session_id}-last-reply-{ts}.wav"
    # Prefer local repo `src/audio-output` folder to store files
    repo_audio_out = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'audio-output'))
    try:
        os.makedirs(repo_audio_out, exist_ok=True)
        return os.path.join(repo_audio_out, file_name)
    except Exception:
        # Fallback to global OUTPUT_DIR if creation fails
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        return os.path.join(OUTPUT_DIR, file_name)

def _generate_inline(text: str, out_path: str, language: str, speaker_wav: str | None, model_name: str):
    """Attempt inline generation using local coqui-tts import."""
    try:
        from TTS.api import TTS  # type: ignore
    except ImportError:
        return False, "ImportError lokalnego coqui-tts"
    try:
        tts = TTS(model_name, progress_bar=False).to("cpu")
        tts.tts_to_file(
            text=text,
            file_path=out_path,
            speaker_wav=speaker_wav,
            language=language
        )
        return True, None
    except Exception as e:
        return False, str(e)

def _generate_via_subprocess(text: str, out_path: str, language: str, speaker_wav: str | None, model_name: str):
    """Fallback: invoke external Python (XTTS_PYTHON) running scripts/xtts_generate.py."""
    python_path = os.getenv('XTTS_PYTHON')
    if not python_path:
        return False, "Brak XTTS_PYTHON – ustaw ścieżkę do Python 3.12 z coqui-tts"
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'xtts_generate.py'))
    if not os.path.exists(script_path):
        return False, f"Brak skryptu pomocniczego: {script_path}"
    cmd = [
        python_path,
        script_path,
        '--text', text,
        '--out', out_path,
        '--language', language,
        '--model', model_name
    ]
    if speaker_wav:
        cmd += ['--speaker-wav', speaker_wav]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if completed.returncode != 0:
            return False, f"Subprocess error: {completed.stderr.strip()}" or completed.stdout.strip()
        return True, None
    except Exception as e:
        return False, str(e)

def generate_audio_from_last_assistant(history: List[Dict], session_id: str, assistant_name: str):
    if not history:
        console.print_error("Brak historii – nie ma czego zamienić na audio.")
        return
    text = _find_last_assistant_reply(history)
    if not text:
        console.print_error("Nie znaleziono ostatniej odpowiedzi asystenta.")
        return

    language = os.getenv('XTTS_LANGUAGE', 'pl')
    model_name = os.getenv('XTTS_MODEL_NAME', 'tts_models/multilingual/multi-dataset/xtts_v2')
    speaker_wav = os.getenv('XTTS_SPEAKER_WAV')
    if speaker_wav and not os.path.exists(speaker_wav):
        console.print_error(f"Plik XTTS_SPEAKER_WAV nie istnieje: {speaker_wav}. Pomijam.")
        speaker_wav = None
    # Default speaker_wav: try repo sample from M2/text-to-speach-xtts
    if not speaker_wav:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        sample_wav = os.path.join(repo_root, 'M2', 'text-to-speach-xtts', 'sample-agent.wav')
        if os.path.exists(sample_wav):
            speaker_wav = sample_wav
            console.print_info(f"Używam domyślnego głosu: {speaker_wav}")

    out_path = _build_output_path(session_id)
    console.print_info(f"Generowanie audio ({language}) z ostatniej odpowiedzi {assistant_name}...")
    console.print_info(f"Docelowy plik: {out_path}")

    success, error = _generate_inline(text, out_path, language, speaker_wav, model_name)
    if not success:
        console.print_info("Inline coqui-tts niedostępne – próba fallback przez XTTS_PYTHON.")
        fb_success, fb_error = _generate_via_subprocess(text, out_path, language, speaker_wav, model_name)
        if not fb_success:
            console.print_error(f"Nie udało się wygenerować audio. Inline: {error}; Fallback: {fb_error}")
            console.print_info("Podaj własny głos w .env: XTTS_SPEAKER_WAV=/ścieżka/voice.wav\n" +
                              "Lub upewnij się, że istnieje plik M2/text-to-speach-xtts/sample-agent.wav.\n" +
                              "Instrukcja (pyenv):\n  pyenv install 3.12.12\n  pyenv virtualenv 3.12.12 xtts312\n  pyenv shell xtts312\n  pip install coqui-tts\n  export XTTS_PYTHON=$(pyenv which python)\n")
            return

    console.print_info(f"Plik audio zapisany: {out_path}")
    console.print_info("Odtwarzanie (macOS): afplay '" + out_path + "'")
