import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # App Settings
    PROJECT_NAME: str = "Vectorless RAG"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # API Keys
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    GEMINI_TIMEOUT_SECONDS: int = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "30"))
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v3.2")
    OPENROUTER_TIMEOUT_SECONDS: int = int(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "60"))


    # File Upload Settings
    UPLOAD_DIR: str = os.path.join(os.getcwd(), "data", "uploads")
    DEFAULT_TREE_PATH: str = os.getenv("DEFAULT_TREE_PATH", "data/hp1_pageindex_tree.json")
    DEFAULT_BOOK_PDF: str = os.getenv(
        "DEFAULT_BOOK_PDF",
        "data/J.K. Rowling - HP 1 - Harry Potter and the Philosophers Stone.pdf",
    )

    class Config:
        case_sensitive = True

settings = Settings()
