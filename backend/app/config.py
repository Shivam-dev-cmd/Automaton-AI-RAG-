import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT_DIR / ".env")

class Settings:
    PROJECT_NAME: str = "Automaton AI RAG Chatbot"
    VERSION: str = "1.0.0"
    
    # Paths to Knowledge Base files
    WORKSPACE_DIR: Path = ROOT_DIR
    KB_JSON_PATH: Path = ROOT_DIR / "automaton_ai_knowledge_base.json"
    KB_MD_PATH: Path = ROOT_DIR / "automaton_ai_knowledge_base.md"
    
    # Server configuration
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", 8000))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")
    
    # LLM Settings
    # Auto-detect provider if keys exist, default to deterministic
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    # Default provider resolution
    _configured_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if _configured_provider in ("gemini", "openai", "deterministic"):
        LLM_PROVIDER: str = _configured_provider
    elif GEMINI_API_KEY:
        LLM_PROVIDER: str = "gemini"
    elif OPENAI_API_KEY:
        LLM_PROVIDER: str = "openai"
    else:
        LLM_PROVIDER: str = "deterministic"
        
    # RAG & Guardrail Settings
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.28"))
    MAX_RETRIEVED_CHUNKS: int = int(os.getenv("MAX_RETRIEVED_CHUNKS", "5"))
    STRICT_GUARDRAILS: bool = os.getenv("STRICT_GUARDRAILS", "True").lower() in ("true", "1", "yes")
    
    # Official Contact Routing
    CONNECT_URL: str = "https://automatonai.com/connect/"
    CAREERS_EMAIL: str = "careers@automatonai.com"
    INFO_EMAIL: str = "info@automatonai.com"
    PHONE_PRIMARY: str = "+91 8698200400"
    OFFICE_LOCATION: str = "Hinjewadi, Pune, Maharashtra, India"

settings = Settings()
