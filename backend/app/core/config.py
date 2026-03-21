import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # App Settings
    PROJECT_NAME: str = "Vectorless RAG"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Google / Gemini (kept for future use)
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # DeepSeek Official API
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_TIMEOUT_SECONDS: int = int(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "120"))

    # File paths
    UPLOAD_DIR: str = os.path.join(os.getcwd(), "data", "uploads")
    DEFAULT_TREE_PATH: str = os.getenv("DEFAULT_TREE_PATH", "data/hp1_pageindex_tree.json")
    DEFAULT_BOOK_PDF: str = os.getenv(
        "DEFAULT_BOOK_PDF",
        "data/J.K. Rowling - HP 1 - Harry Potter and the Philosophers Stone.pdf",
    )

    class Config:
        case_sensitive = True

settings = Settings()
