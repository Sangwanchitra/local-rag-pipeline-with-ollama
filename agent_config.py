# agent_config.py

# Current Agent Configuration
PROVIDER = "Ollama"
BASE_URL = "http://localhost:11434"
MODEL_NAME = "llama3:8b"

# Ensure cloud-based providers are explicitly disabled
USE_CLOUD_PROVIDER = False

def get_agent_config():
    """Returns the current agent configuration."""
    return {
        "provider": PROVIDER,
        "base_url": BASE_URL,
        "model": MODEL_NAME,
        "use_cloud_provider": USE_CLOUD_PROVIDER
    }
