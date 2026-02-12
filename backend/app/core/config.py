"""
Core configuration settings for the RAG API
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import ClassVar

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Calculate project root dynamically
    PROJECT_ROOT: ClassVar[Path] = Path(__file__).parent.parent.parent
    # API Settings
    app_name: str = "UET Lahore RAG API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # OpenAI Settings
    openai_api_key: str
    embedding_model: str = "text-embedding-3-large"
    llm_model: str = "gpt-4o"
    filtering_model: str = "gpt-4o-mini"
    
    # Data Paths (Relative to project root)
    data_file: str = str(PROJECT_ROOT / "data" / "processed" / "rag_optimized_data.txt")
    chroma_db_path: str = str(PROJECT_ROOT / "chroma_db_openai")
    
    # Retrieval Settings
    top_k: int = 6
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()