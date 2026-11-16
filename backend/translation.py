import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


class TranslationError(RuntimeError):
    """Raised when the Groq translation request cannot produce a translation."""


class TranslationConfigurationError(TranslationError):
    """Raised when English translation is requested without a configured API key."""


def translate_to_english(text: str) -> str:
    if not re.search(r"[\u0600-\u06FF]", text):
        return text

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise TranslationConfigurationError("GROQ_API_KEY is not configured.")

    try:
        response = requests.post(
            GROQ_CHAT_COMPLETIONS_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Translate the Urdu text into clear, faithful English. "
                            "Preserve Arabic quotations, names, citations, honorifics, "
                            "and paragraph structure. Do not add commentary."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                "temperature": 0.1,
            },
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise TranslationError("The Groq translation request failed.") from exc

    try:
        translation = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise TranslationError("The Groq response did not contain a translation.") from exc

    if not isinstance(translation, str) or not translation.strip():
        raise TranslationError("The Groq response did not contain a translation.")

    return translation.strip()