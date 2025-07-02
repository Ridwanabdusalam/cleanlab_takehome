"""
Central configuration for the trustworthiness detector.
All model and API settings are managed here.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Default model - can be overridden by DEFAULT_MODEL env var
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini/gemini-pro")

# API key validation
def get_api_key():
    """Get the appropriate API key based on the model being used."""
    # Check which API keys are available
    api_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
    }
    
    # Return first available key
    for key_name, key_value in api_keys.items():
        if key_value:
            return key_name, key_value
    
    return None, None

# Model provider mapping
MODEL_PROVIDERS = {
    "gpt-3.5-turbo": "OPENAI_API_KEY",
    "gpt-4": "OPENAI_API_KEY",
    "claude-2": "ANTHROPIC_API_KEY",
    "claude-3-opus": "ANTHROPIC_API_KEY",
    "gemini/gemini-pro": "GEMINI_API_KEY",
    "gemini/gemini-1.5-pro": "GEMINI_API_KEY",
}

def validate_model_api_key(model: str = None):
    """Check if the appropriate API key is set for the given model."""
    model = model or DEFAULT_MODEL
    
    if model in MODEL_PROVIDERS:
        required_key = MODEL_PROVIDERS[model]
        if not os.getenv(required_key):
            available_keys = [k for k in ["OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"] 
                            if os.getenv(k)]
            if available_keys:
                return False, f"Model {model} requires {required_key}, but only {available_keys} found. Update DEFAULT_MODEL in .env"
            else:
                return False, f"No API keys found. Please set {required_key} in your .env file"
    
    return True, "API key validated"
