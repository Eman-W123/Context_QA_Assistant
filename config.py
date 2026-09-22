import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

# Load environment variables from .env file
load_dotenv()

# --- Configuration Constants ---
DEFAULT_MODEL_PROVIDER = "gemini"  # Options: "gemini", "ollama"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OLLAMA_MODEL = "llama3"

# Token budget threshold for context engineering (summarization trigger)
TOKEN_BUDGET = 2000


def get_model(provider: str = DEFAULT_MODEL_PROVIDER, model_name: str = None, temperature: float = 0.7):
    """
    Factory function returning a LangChain Chat Model instance 
    based on the selected provider and model name.
    """
    provider_clean = provider.lower().strip()

    if provider_clean == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is missing from environment variables or .env file.")
        
        selected_model = model_name if model_name else DEFAULT_GEMINI_MODEL
        
        return ChatGoogleGenerativeAI(
            model=selected_model,
            google_api_key=api_key,
            temperature=temperature,
            streaming=True
        )

    elif provider_clean == "ollama":
        selected_model = model_name if model_name else DEFAULT_OLLAMA_MODEL
        
        return ChatOllama(
            model=selected_model,
            temperature=temperature
        )

    else:
        raise ValueError(f"Unsupported model provider '{provider}'. Options: 'gemini', 'ollama'")