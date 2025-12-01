import uuid
from typing import List, Any, Union
import os
from files import session_files
from files.wal import append_to_wal
from llm.gemini_client import GeminiLLMClient
from llm.llama_client import LlamaClient
from assistant import Assistant
from cli import console

# Context token limit

# Engine to Client Class mapping
ENGINE_MAPPING = {
    'LLAMA_CPP': LlamaClient,
    'GEMINI': GeminiLLMClient,
}


class ChatSession:
    """
    Manages everything related to a single chat session.
    Encapsulates session ID, conversation history, assistant, and LLM chat session.
    """

    def __init__(self, assistant: Assistant, session_id: str | None = None, history: List[Any] | None = None, title: str | None = None):
        """
        Initialize a chat session.

        Args:
            assistant: Assistant instance that defines the behavior and model for this session
            session_id: Unique session identifier. If None, generates a new UUID.
            history: Initial conversation history. If None, starts empty.
        """
        self.assistant = assistant
        self.session_id = session_id or str(uuid.uuid4())
        self._history = history or []
        self._llm_client: Union[GeminiLLMClient, LlamaClient, None] = None
        self._llm_chat_session = None
        self._max_context_tokens = 32768
        self._title: str | None = title
        self._initialize_llm_session()

    def _initialize_llm_session(self):
        """
        Creates or recreates the LLM chat session with current history.
        This should be called after any history modification.
        """
        # Walidacja zmiennej ENGINE
        engine = os.getenv('ENGINE', 'GEMINI').upper()
        if engine not in ENGINE_MAPPING:
            valid_engines = ', '.join(ENGINE_MAPPING.keys())
            raise ValueError(f"ENGINE musi być jedną z wartości: {valid_engines}, otrzymano: {engine}")

        # Initialize LLM client if not already created
        if self._llm_client is None:
            SelectedClientClass = ENGINE_MAPPING.get(engine, GeminiLLMClient)
            console.print_info(SelectedClientClass.preparing_for_use_message())
            self._llm_client = SelectedClientClass.from_environment()
            console.print_info(self._llm_client.ready_for_use_message())

        self._llm_chat_session = self._llm_client.create_chat_session(
            system_instruction=self.assistant.system_prompt,
            history=self._history,
            thinking_budget=0
        )


    @classmethod
    def load_from_file(cls, session_id: str) -> tuple['ChatSession | None', str | None]:
        """
        Loads a session from disk creating Assistant from stored metadata.

        Returns:
            (ChatSession | None, error_message | None)
        """
        history, system_role, assistant_name, title, error = session_files.load_session_history(session_id)
        if error:
            return None, error
        # If metadata missing fallback to generic AZOR assistant name and role
        role = system_role or "Jesteś pomocnym asystentem."
        name = assistant_name or "AZOR"
        assistant = Assistant(system_prompt=role, name=name)
        session = cls(assistant=assistant, session_id=session_id, history=history, title=title)
        return session, None

    def save_to_file(self) -> tuple[bool, str | None]:
        """
        Saves this session to disk.
        Only saves if history has at least one complete exchange.

        Returns:
            tuple: (success: bool, error_message: str | None)
        """
        # Sync history from LLM session before saving
        if self._llm_chat_session:
            self._history = self._llm_chat_session.get_history()

        return session_files.save_session_history(
            self.session_id,
            self._history,
            self.assistant.system_prompt,
            self.assistant.name,
            self._llm_client.get_model_name(),
            self._title
        )

    def send_message(self, text: str):
        """
        Sends a message to the LLM and returns the response.
        Updates internal history automatically and logs to WAL.

        Args:
            text: User's message

        Returns:
            Response object from Google GenAI
        """
        if not self._llm_chat_session:
            raise RuntimeError("LLM session not initialized")

        original_text = text

        # If no title yet and this is the very first user message, augment request
        is_first_user_message = len(self.get_history()) == 0
        if is_first_user_message and not self._title:
            augmented = (
                "Na potrzeby aplikacji zwróć odpowiedź w formacie JSON z dwoma polami: "
                "title (krótki, maks 6 słów, bez cudzysłowów dodatkowych) oraz answer (normalna odpowiedź). "
                "Tylko czysty JSON, bez komentarzy ani dodatkowego tekstu przed czy po. "
                "Oto oryginalna wiadomość użytkownika:\n" + original_text
            )
            text = augmented

        response = self._llm_chat_session.send_message(text)

        # Sync history after message
        self._history = self._llm_chat_session.get_history()

        # If we augmented the first user message, restore original user text in history for display
        if is_first_user_message and not self._title:
            hist = self._history
            # After send_message, last two entries should be user then assistant
            if len(hist) >= 2 and hist[-2]['role'] == 'user':
                hist[-2]['parts'][0]['text'] = original_text
                self._history = hist

        # Attempt to parse title from first response if augmented
        if is_first_user_message and not self._title:
            parsed_title = self._extract_title_from_response(response.text)
            if not parsed_title:
                # Fallback heuristic: truncate original prompt
                parsed_title = (original_text.strip()[:60]).replace('\n', ' ')
            self._title = parsed_title.strip()
        else:
            # For subsequent messages: if model still returns JSON {title, answer}, strip to answer only
            cleaned = self._extract_answer_only(response.text)
            if cleaned is not None:
                # Replace assistant last message text with cleaned answer
                hist = self.get_history()
                if hist and hist[-1]['role'] in ('model', 'assistant'):
                    hist[-1]['parts'][0]['text'] = cleaned
                    self._history = hist

        # Log to WAL
        total_tokens = self.count_tokens()
        success, error = append_to_wal(
            session_id=self.session_id,
            prompt=original_text,
            response_text=response.text,
            total_tokens=total_tokens,
            model_name=self._llm_client.get_model_name()
        )

        if not success and error:
            # We don't want to fail the entire message sending because of WAL issues
            # Just log the error to stderr or similar - but for now we'll silently continue
            pass

        return response

    def get_history(self) -> List[Any]:
        """Returns the current conversation history."""
        # Return internal history to preserve local adjustments (e.g., restored first user message)
        return self._history

    def get_title(self) -> str | None:
        return self._title

    def rename_title(self, new_title: str) -> tuple[bool, str | None]:
        if not new_title.strip():
            return False, "Tytuł nie może być pusty."
        self._title = new_title.strip()
        success, error = self.save_to_file()
        if not success:
            return False, error or "Błąd zapisu po zmianie tytułu."
        return True, None

    def clear_history(self):
        """Clears all conversation history and reinitializes the LLM session."""
        self._history = []
        self._initialize_llm_session()
        self.save_to_file()

    def pop_last_exchange(self) -> bool:
        """
        Removes the last user-assistant exchange from history.

        Returns:
            bool: True if successful, False if insufficient history
        """
        current_history = self.get_history()

        if len(current_history) < 2:
            return False

        # Remove last 2 entries (user + assistant)
        self._history = current_history[:-2]

        # Reinitialize LLM session with modified history
        self._initialize_llm_session()

        self.save_to_file()

        return True

    def count_tokens(self) -> int:
        """
        Counts total tokens in the conversation history.

        Returns:
            int: Total token count
        """
        if not self._llm_client:
            return 0
        return self._llm_client.count_history_tokens(self._history)

    def is_empty(self) -> bool:
        """
        Checks if session has any complete exchanges.

        Returns:
            bool: True if history has less than 2 entries
        """
        return len(self._history) < 2

    def get_remaining_tokens(self) -> int:
        """
        Calculates remaining tokens based on context limit.

        Returns:
            int: Remaining token count
        """
        total = self.count_tokens()
        return self._max_context_tokens - total

    def get_token_info(self) -> tuple[int, int, int]:
        """
        Gets comprehensive token information for this session.

        Returns:
            tuple: (total_tokens, remaining_tokens, max_tokens)
        """
        total_tokens = self.count_tokens()
        remaining_tokens = self._max_context_tokens - total_tokens
        max_tokens = self._max_context_tokens
        return total_tokens, remaining_tokens, max_tokens

    @property
    def assistant_name(self) -> str:
        """
        Gets the display name of the assistant.

        Returns:
            str: The assistant's display name
        """
        return self.assistant.name

    def set_assistant(self, assistant: Assistant):
        """
        Updates the assistant for this session and reinitializes the LLM session
        with the existing history.
        """
        self.assistant = assistant
        self._initialize_llm_session()

    def _extract_title_from_response(self, response_text: str) -> str | None:
        """Try to parse JSON and return title field."""
        import json
        txt = response_text.strip()
        # Sometimes model might wrap JSON in markdown fences
        if txt.startswith('```'):
            # remove first and last fence if present
            lines = txt.splitlines()
            if lines[0].startswith('```'):
                # find last fence
                try:
                    last_index = next(i for i,l in enumerate(lines[1:], start=1) if l.startswith('```'))
                    txt = '\n'.join(lines[1:last_index])
                except StopIteration:
                    txt = '\n'.join(lines[1:])
        if not (txt.startswith('{') and txt.endswith('}')):
            return None
        try:
            data = json.loads(txt)
            title = data.get('title')
            answer = data.get('answer')
            if title and answer:
                # Replace response text in history with answer only (remove JSON noise)
                # Last entry should be assistant/model role
                hist = self.get_history()
                if hist and hist[-1]['role'] in ('model', 'assistant'):
                    hist[-1]['parts'][0]['text'] = answer
                    self._history = hist
                return title
        except Exception:
            return None
        return None

    def _extract_answer_only(self, response_text: str) -> str | None:
        """If response is a JSON with fields {title, answer}, return only answer; else None."""
        import json
        txt = response_text.strip()
        if txt.startswith('```'):
            lines = txt.splitlines()
            if lines[0].startswith('```'):
                try:
                    last_index = next(i for i,l in enumerate(lines[1:], start=1) if l.startswith('```'))
                    txt = '\n'.join(lines[1:last_index])
                except StopIteration:
                    txt = '\n'.join(lines[1:])
        if not (txt.startswith('{') and txt.endswith('}')):
            return None
        try:
            data = json.loads(txt)
            answer = data.get('answer')
            title = data.get('title')
            if answer and title:
                return answer
        except Exception:
            return None
        return None