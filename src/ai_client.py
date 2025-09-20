# src/ai_client.py
import os
from openai import OpenAI

def get_openai_client() -> OpenAI:
    """
    Returns an OpenAI client using the OPENAI_API_KEY environment variable.
    Fails fast with a helpful message if the key isn't set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in your environment first.\n"
            "Windows (PowerShell):  [Environment]::SetEnvironmentVariable('OPENAI_API_KEY','sk-...','User')\n"
            "macOS/Linux (bash/zsh): export OPENAI_API_KEY='sk-...'"
        )
    return OpenAI(api_key=api_key)
