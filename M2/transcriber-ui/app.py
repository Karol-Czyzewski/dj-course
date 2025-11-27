import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
import pyaudio
import wave
import os
import time
import threading
import queue
import sys
import logging
import logging.handlers
from typing import TextIO

# --- Global Configuration ---
APP_TITLE = "Azor Transcriber"
# Set to True to print output to the console (standard output/stderr).
VERBOSE = False
# Explicit project paths (user request)
PROJECT_BASE_OVERRIDE = "/Users/karol/dev/dj-course/M2/transcriber-ui"

BASE_DIR = PROJECT_BASE_OVERRIDE if os.path.isdir(PROJECT_BASE_OVERRIDE) else os.path.dirname(os.path.abspath(__file__))
LOG_FILENAME = os.path.join(BASE_DIR, "transcriber.log")
AUDIO_DIR = os.path.join(BASE_DIR, "audio-recordings")
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts-data")

# --- Logging Setup ---
class StreamToLogger(TextIO):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    This captures stdout/stderr, including print() statements.
    """
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        # Handle buffer and write line by line
        for line in buf.rstrip().splitlines():
            # Check if the line is not empty (prevents logging empty lines from print())
            if line.strip():
                self.logger.log(self.level, line.strip())

    def flush(self):
        # Required by TextIO interface, but we flush line-by-line in write
        pass

# Configure the global logger BEFORE application startup
def setup_logging():
    """Configures logging to file and optionally console."""
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(PROMPTS_DIR, exist_ok=True)
    
    # 1. Root logger setup
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO) # Capture everything from INFO level up

    # 2. File Handler (Always active)
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILENAME, 
        maxBytes=1024*1024*5, # 5 MB per file
        backupCount=5,
        encoding='utf-8'
    )
    # Define a simple formatter for the file
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # 3. Console Handler (Only active if VERBOSE is True)
    if VERBOSE:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # 4. Redirect stdout and stderr to the logger
    sys.stdout = StreamToLogger(root_logger, logging.INFO)
    sys.stderr = StreamToLogger(root_logger, logging.ERROR)

setup_logging()
logging.info("Application initialization started.")

# --- Whisper Dependencies ---
# Ensure you have installed: pip install torch transformers librosa
# (Librosa might require ffmpeg)
try:
    import torch
    from transformers import pipeline
except ImportError:
    logging.error("ERROR: 'transformers' or 'torch' libraries not found.")
    logging.error("Install them using: pip install torch transformers")
    exit()

# === 1. Transcription Configuration ===
MODEL_NAME = "openai/whisper-tiny"

def generate_ids() -> tuple[str, str]:
    """Generates a base ID and returns audio and prompt filenames."""
    ts = int(time.time())
    base = f"recording-{ts}"
    audio_path = os.path.join(AUDIO_DIR, f"{base}.wav")
    prompt_path = os.path.join(PROMPTS_DIR, f"{base}.json")
    return audio_path, prompt_path

def transcribe_audio(audio_path: str, model_name: str) -> str:
    """
    Loads the Whisper model and transcribes the audio file.
    This function is blocking and should be run in a separate thread.
    """
    try:
        logging.info(f"Loading model: {model_name}...")
        # Initialize pipeline
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")
        
        asr_pipeline = pipeline(
            "automatic-speech-recognition", 
            model=model_name,
            device=device
        )

        logging.info(f"Starting transcription for file: {audio_path}...")
        result = asr_pipeline(audio_path)
        
        transcription = result["text"].strip()
        
        logging.info("Transcription finished.")
        return transcription

    except FileNotFoundError:
        logging.error(f"ERROR: Audio file not found at path: {audio_path}")
        return f"ERROR: Audio file not found at path: {audio_path}"
    except Exception as e:
        logging.error(f"An unexpected error occurred during transcription: {e}", exc_info=True)
        return f"An unexpected error occurred during transcription: {e}"


# === 2. Recording Configuration ===
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Standard for speech models (Whisper)
MAX_RECORD_DURATION = 30 # Maximum recording length in seconds

# === 3. Tkinter GUI Application ===
class AudioRecorderApp:
    def __init__(self, master):
        self.master = master
        
        # 1. Set application title (window title)
        master.title(APP_TITLE)
        
        # 2. Set the application name for the OS/taskbar
        # This is cross-platform attempt to set the application name
        try:
            # For macOS and some X11 environments
            self.master.tk.call('wm', 'iconname', self.master._w, APP_TITLE)
        except tk.TclError:
            # Standard method, usually works on Windows/Linux
            self.master.wm_iconname(APP_TITLE)
            
        master.geometry("600x450") # Slightly larger window
        master.config(bg="#121212") # Set dark background for root

        # --- TKINTER WIDGET STYLES (ttk) ---
        style = ttk.Style()
        style.theme_use('default') 

        # Configure the dark background for the Notebook tabs
        style.configure('TNotebook', background='#121212', borderwidth=0)
        style.configure('TNotebook.Tab', background='#1E1E1E', foreground='white', borderwidth=0)
        style.map('TNotebook.Tab', background=[('selected', '#0F0F0F')], foreground=[('selected', 'white')])

        # 1. Define new style for dark gray buttons
        style.configure('Dark.TButton',
                        background='#333333',    
                        foreground='white',     
                        font=('Arial', 14),
                        bordercolor='#333333',
                        borderwidth=0,
                        focuscolor='#333333',
                        padding=(20, 10, 20, 10) 
                       )
        
        # 2. Define button appearance in different states (active/disabled)
        style.map('Dark.TButton',
                  background=[('active', '#555555'), # Lighter gray for hover/active state
                              ('disabled', '#333333')], # Disabled state uses the default background
                 )

        logging.info("GUI initialization started.")

        # Initialize PyAudio
        try:
            self.p = pyaudio.PyAudio()
        except Exception as e:
            logging.critical(f"Could not initialize PyAudio: {e}. Destroying GUI.")
            messagebox.showerror("PyAudio Error", f"Could not initialize PyAudio: {e}\nDo you have 'portaudio' installed?")
            master.destroy()
            return
            
        self.frames = []
        self.stream = None
        self.recording = False
        self.start_time = None
        self.record_timer_id = None 

        # Queue for inter-thread communication
        self.transcription_queue = queue.Queue()
        
        # --- TAB MENU SETUP (Notebook) ---
        self.notebook = ttk.Notebook(master, style='TNotebook')
        self.notebook.pack(pady=10, padx=10, fill='both', expand=True)

        # 1. Transcriber Tab
        self.transcriber_frame = tk.Frame(self.notebook, bg="#121212") # Set dark background for frame
        self.notebook.add(self.transcriber_frame, text='Transcriber')

        # 2. History Tab
        self.history_frame = tk.Frame(self.notebook, bg="#121212")
        self.notebook.add(self.history_frame, text='History')

        # History UI: list of transcriptions + preview + delete
        list_frame = tk.Frame(self.history_frame, bg="#121212")
        list_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        tk.Label(list_frame, text="Saved transcriptions:", font=('Arial', 12, 'bold'), fg='white', bg="#121212").pack(anchor='w')

        # Container for list + scrollbar to prevent it from covering buttons
        listbox_container = tk.Frame(list_frame, bg="#121212")
        listbox_container.pack(fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(listbox_container, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.history_list = tk.Listbox(listbox_container, height=14, bg='#1E1E1E', fg='white', yscrollcommand=scrollbar.set)
        self.history_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.history_list.yview)
        self.history_list.bind('<<ListboxSelect>>', self.on_history_select)

        # Buttons stay below the list; list doesn't overtake space thanks to its container
        btn_frame = tk.Frame(list_frame, bg="#121212")
        btn_frame.pack(fill=tk.X, pady=(10,0))
        # CTA: Delete selected
        self.delete_button = ttk.Button(btn_frame, text="Delete", command=self.delete_selected, style='Dark.TButton')
        self.delete_button.pack(fill=tk.X)

        # CTA: Delete all
        self.delete_all_button = ttk.Button(btn_frame, text="Delete All", command=self.delete_all, style='Dark.TButton')
        self.delete_all_button.pack(fill=tk.X, pady=(8, 0))

        preview_frame = tk.Frame(self.history_frame, bg="#121212")
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        tk.Label(preview_frame, text="Preview:", font=('Arial', 12, 'bold'), fg='white', bg="#121212").pack(anchor='w')
        self.history_display = tk.Text(preview_frame,
                           height=20,
                           wrap=tk.WORD,
                           font=('Arial', 11),
                           relief=tk.SUNKEN,
                           bg='#1E1E1E',
                           fg='white',
                           insertbackground='white',
                           state=tk.DISABLED)
        self.history_display.pack(fill=tk.BOTH, expand=True)


        # 3. Settings Tab
        self.settings_frame = tk.Frame(self.notebook, bg="#121212") 
        self.notebook.add(self.settings_frame, text='Settings')

        # Content for Settings Tab
        tk.Label(self.settings_frame, text="Under construction...", font=('Arial', 18), fg='gray', bg="#121212").pack(pady=50)


        # --- Transcriber Tab Elements ---
        
        # Record Button
        self.record_button = ttk.Button(self.transcriber_frame, 
                                        text="Record", 
                                        command=self.toggle_recording, 
                                        style='Dark.TButton')
        self.record_button.pack(pady=20, fill=tk.X, padx=20) 

        # Transcribed Text Display (Read-only Text widget)
        self.transcription_display = tk.Text(self.transcriber_frame, 
                                             height=10, 
                                             wrap=tk.WORD, 
                                             font=('Arial', 11),
                                             relief=tk.SUNKEN, 
                                             bg='#1E1E1E', 
                                             fg='white', 
                                             insertbackground='white', 
                                             state=tk.DISABLED 
                                             )
        self.transcription_display.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        # Initial text insertion for tk.Text
        self.transcription_display.config(state=tk.NORMAL)
        self.transcription_display.insert(tk.END, "Transcribed text will appear here. Select it to copy.")
        self.transcription_display.config(state=tk.DISABLED)


        # Exit Button
        self.exit_button = ttk.Button(master, 
                                      text="Exit", 
                                      command=self.on_closing,
                                      style='Dark.TButton')
        self.exit_button.pack(pady=10)

        # Handle window closing
        master.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Start the loop checking the queue
        self.master.after(100, self.check_transcription_queue)
        logging.info("GUI initialized successfully.")
    
    def copy_to_clipboard(self, text: str):
        """Copies the given text to the system clipboard."""
        self.master.clipboard_clear()
        self.master.clipboard_append(text)
        logging.info("Transcription copied to clipboard.")

    def toggle_recording(self):
        """Toggles the recording state (start/stop)."""
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        """Starts the audio recording process."""
        self.recording = True
        self.frames = []
        self.start_time = time.time()
        logging.info("Recording started.")
        
        try:
            self.stream = self.p.open(format=FORMAT,
                                     channels=CHANNELS,
                                     rate=RATE,
                                     input=True,
                                     frames_per_buffer=CHUNK)

            # Update button text to show status
            self.record_button.config(text="Stop Recording") 
            
            # Update text display
            self.transcription_display.config(state=tk.NORMAL)
            self.transcription_display.delete('1.0', tk.END)
            self.transcription_display.insert(tk.END, "Recording in progress... (max 30s)")
            self.transcription_display.config(state=tk.DISABLED)
            
            self.read_chunk()
            # Set a timer for automatic stop
            self.record_timer_id = self.master.after(MAX_RECORD_DURATION * 1000, self.auto_stop_recording)

        except Exception as e:
            self.recording = False
            self.record_button.config(text="Record", state=tk.NORMAL) 
            logging.error(f"Microphone stream error on start: {e}")
            messagebox.showerror("Audio Error", f"Could not open microphone stream: {e}\nCheck your microphone connection and permissions.")
            if self.record_timer_id:
                self.master.after_cancel(self.record_timer_id)
                self.record_timer_id = None
            
    def read_chunk(self):
        """Reads one audio chunk and schedules the next call."""
        if self.recording:
            try:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                self.frames.append(data)
                self.master.after(1, self.read_chunk) 
            except IOError as e:
                logging.error(f"Stream read IOError: {e}")
                self.stop_recording()

    def auto_stop_recording(self):
        """Automatically stops recording after MAX_RECORD_DURATION expires."""
        if self.recording:
            logging.info(f"Automatic stop triggered after {MAX_RECORD_DURATION} seconds.")
            self.stop_recording()
            messagebox.showinfo("Recording Finished", f"The recording was stopped automatically after {MAX_RECORD_DURATION} seconds. Starting transcription...")

    def stop_recording(self):
        """Stops the stream, saves the file, and starts the transcription thread."""
        if not self.recording:
            return

        self.recording = False
        
        if self.record_timer_id:
            self.master.after_cancel(self.record_timer_id)
            self.record_timer_id = None

        # Stop and close the stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        logging.info("Audio stream closed.")

        AUDIO_OUTPUT_FILENAME, PROMPT_JSON_FILENAME = generate_ids()
        
        # Update button status for user feedback
        self.record_button.config(text="Saving...", state=tk.DISABLED) 
        self.master.update_idletasks()

        # Save to WAVE file
        try:
            with wave.open(AUDIO_OUTPUT_FILENAME, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(self.frames))
            logging.info(f"File saved successfully to {AUDIO_OUTPUT_FILENAME}")
            
            self.record_button.config(text="Transcribing...")
            
            # Update text in read-only Text widget
            self.transcription_display.config(state=tk.NORMAL)
            self.transcription_display.delete('1.0', tk.END)
            self.transcription_display.insert(tk.END, "Transcription in progress (this may take a while)...")
            self.transcription_display.config(state=tk.DISABLED)
            
            # === START TRANSCRIPTION IN A THREAD ===
            transcription_thread = threading.Thread(
                target=self.run_transcription,
                args=(AUDIO_OUTPUT_FILENAME, PROMPT_JSON_FILENAME,),
                daemon=True
            )
            transcription_thread.start()
            logging.info("Transcription thread started.")

        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save WAVE file: {e}")
            self.record_button.config(text="Record", state=tk.NORMAL) 
            logging.error(f"Error saving wave file: {e}", exc_info=True)

    def run_transcription(self, audio_path, prompt_json_path):
        """
        Method executed in a separate thread. 
        Calls transcription and puts the result in the queue.
        """
        logging.info(f"Running transcription for {audio_path} in thread: {threading.get_ident()}")
        transcription = transcribe_audio(audio_path, MODEL_NAME)
        # Persist JSON metadata
        try:
            meta = {
                "id": os.path.splitext(os.path.basename(prompt_json_path))[0],
                "audio_path": audio_path,
                "model": MODEL_NAME,
                "timestamp": int(time.time()),
                "text": transcription,
            }
            with open(prompt_json_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(meta, f, ensure_ascii=False, indent=2)
            logging.info(f"Saved transcription JSON to {prompt_json_path}")
        except Exception as e:
            logging.error(f"Failed to save JSON metadata: {e}")
        self.transcription_queue.put((transcription, prompt_json_path))

    def check_transcription_queue(self):
        """
        Checks the queue for transcription results.
        Run in the main GUI thread.
        """
        try:
            result = self.transcription_queue.get(block=False)
            
            # 1. Update Transcriber tab (main output)
            text_result, prompt_json_path = result if isinstance(result, tuple) else (result, None)
            self.transcription_display.config(state=tk.NORMAL)
            self.transcription_display.delete('1.0', tk.END)
            self.transcription_display.insert(tk.END, text_result)
            self.transcription_display.config(state=tk.DISABLED)
            
            # 2. Update History tab (last output)
            # Refresh history list and preview
            self.refresh_history_list()
            if prompt_json_path:
                self.load_history_item(prompt_json_path)
            
            if "ERROR" in text_result:
                logging.warning("Transcription failed with error message.")
                messagebox.showerror("Transcription Failed", "Transcription returned an error. Check logs for details.")
            else:
                # Copy to clipboard upon successful transcription
                self.copy_to_clipboard(text_result) 
                
            self.record_button.config(text="Record", state=tk.NORMAL) # Return to normal state

        except queue.Empty:
            pass
        finally:
            self.master.after(100, self.check_transcription_queue)

    def refresh_history_list(self):
        """Loads JSON files from PROMPTS_DIR into the listbox."""
        try:
            items = sorted([
                f for f in os.listdir(PROMPTS_DIR)
                if f.endswith('.json')
            ])
            self.history_list.delete(0, tk.END)
            for f in items:
                self.history_list.insert(tk.END, f)
        except Exception as e:
            logging.error(f"Failed to refresh history list: {e}")

    def on_history_select(self, event=None):
        sel = self.history_list.curselection()
        if not sel:
            return
        fname = self.history_list.get(sel[0])
        path = os.path.join(PROMPTS_DIR, fname)
        self.load_history_item(path)

    def load_history_item(self, json_path: str):
        """Loads JSON and shows preview."""
        try:
            import json
            with open(json_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            preview = f"ID: {meta.get('id')}\nModel: {meta.get('model')}\nAudio: {meta.get('audio_path')}\nTimestamp: {meta.get('timestamp')}\n\n{meta.get('text','')}"
            self.history_display.config(state=tk.NORMAL)
            self.history_display.delete('1.0', tk.END)
            self.history_display.insert(tk.END, preview)
            self.history_display.config(state=tk.DISABLED)
        except Exception as e:
            logging.error(f"Failed to load history item {json_path}: {e}")

    def delete_selected(self):
        """Deletes selected transcription JSON and its audio file."""
        sel = self.history_list.curselection()
        if not sel:
            messagebox.showinfo("Usuń", "Wybierz element do usunięcia.")
            return
        fname = self.history_list.get(sel[0])
        json_path = os.path.join(PROMPTS_DIR, fname)
        try:
            import json
            with open(json_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            audio_path = meta.get('audio_path')
        except Exception:
            audio_path = None

        if not messagebox.askyesno("Potwierdź usunięcie", f"Usunąć {fname} oraz jego plik audio?"):
            return
        try:
            if os.path.exists(json_path):
                os.remove(json_path)
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
            logging.info(f"Deleted {json_path} and {audio_path}")
            self.refresh_history_list()
            self.history_display.config(state=tk.NORMAL)
            self.history_display.delete('1.0', tk.END)
            self.history_display.config(state=tk.DISABLED)
        except Exception as e:
            logging.error(f"Failed to delete files: {e}")
            messagebox.showerror("Usuwanie nieudane", f"Nie można usunąć plików: {e}")

    def delete_all(self):
        """Deletes all transcription JSON files and their associated audio files."""
        try:
            items = [
                os.path.join(PROMPTS_DIR, f)
                for f in os.listdir(PROMPTS_DIR)
                if f.endswith('.json')
            ]
        except Exception as e:
            logging.error(f"Failed to list history for delete all: {e}")
            messagebox.showerror("Usuwanie nieudane", f"Nie można odczytać listy: {e}")
            return

        if not items:
            messagebox.showinfo("Usuń wszystko", "Brak zapisów do usunięcia.")
            return

        if not messagebox.askyesno("Potwierdź usunięcie", "Usunąć wszystkie zapisy oraz powiązane pliki audio?"):
            return

        removed = 0
        for json_path in items:
            audio_path = None
            try:
                import json
                with open(json_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                audio_path = meta.get('audio_path')
            except Exception:
                audio_path = None

            try:
                if os.path.exists(json_path):
                    os.remove(json_path)
                    removed += 1
                if audio_path and os.path.exists(audio_path):
                    os.remove(audio_path)
            except Exception as e:
                logging.error(f"Failed to delete {json_path} or audio {audio_path}: {e}")

        logging.info(f"Deleted {removed} JSON files and their audio counterparts where available.")
        self.refresh_history_list()
        self.history_display.config(state=tk.NORMAL)
        self.history_display.delete('1.0', tk.END)
        self.history_display.config(state=tk.DISABLED)
        messagebox.showinfo("Usuń wszystko", f"Usunięto {removed} zapisów.")

    def on_closing(self):
        """Handles clean application shutdown."""
        logging.info("Closing application...")
        if self.recording:
            self.stop_recording() 
        
        # Terminate PyAudio
        if self.p:
            self.p.terminate()
        
        self.master.destroy()
        logging.info("Application destroyed.")

# --- Application Startup ---
if __name__ == "__main__":
    logging.info("Whisper model loading might take a moment on first launch...")
    root = tk.Tk()
    app = AudioRecorderApp(root)
    # Initial history load
    app.refresh_history_list()
    root.mainloop()
