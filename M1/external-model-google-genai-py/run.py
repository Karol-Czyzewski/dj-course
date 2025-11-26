from google import genai
from google.genai import types
import os

from dotenv import load_dotenv
load_dotenv()

# if set, print first 4 chars and last 4 chars and dots inside, else print NOT SET
print(f"env var \"GEMINI_API_KEY\" is: { os.getenv('GEMINI_API_KEY', '')[:4] + '...' + os.getenv('GEMINI_API_KEY', '')[-4:] if len(os.getenv('GEMINI_API_KEY', '')) > 0 else 'NOT SET' }")
if not os.getenv('GEMINI_API_KEY'):
    raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it to your Google Gemini API key.")

client = genai.Client()

model = "gemini-2.5-flash"
# model = "gemini-1.5-pro"

# system_role = "you were Gandalf the Grey in the Lord of the Rings. You answer in max 15 words. Your answers are mysterious and magical."

system_role = "jesteś pomocnym asystentem, który odpowiada na pytania w języku polskim i mówi wierszem."


conversation_history = [
    types.Content(
        role="user",
        # parts=[types.Part.from_text(text="What is the best time for coffee?")]
        parts=[types.Part.from_text(text="Jaka jest najlepsza pora na kawę?")]
    ),
    types.Content(
        role="model",
        parts=[types.Part.from_text(text="Najlepsza pora na kawę to rano, mój uczniu.")]
    ),
    types.Content(
        role="user",
        parts=[types.Part.from_text(text="A herbata?")]
    ),
]

response = client.models.generate_content(
    model=model,
    contents=conversation_history,
    config=types.GenerateContentConfig(
        system_instruction=system_role,
        thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
    ),
)

print(response.text)