"""
Configuration settings for the AI Research Assistant
"""
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import Optional, List
from functools import lru_cache
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Keys (support multiple naming conventions)
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None  # Alternative name for Google API
    tavily_api_key: Optional[str] = None
    megallm_api_key: Optional[str] = None  # MegaLLM API key
    
    # MegaLLM Configuration (OpenAI-compatible API)
    megallm_base_url: str = "https://ai.megallm.io/v1"
    megallm_model: str = "llama3-8b-instruct"  # Free tier model
    
    # Model Configuration
    llm_provider: str = "megallm"  # "openai", "google", or "megallm"
    openai_model: str = "gpt-4o"
    google_model: str = "gemini-1.5-flash"
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    # Vector DB Configuration
    faiss_index_path: str = "data/faiss_index"
    pdf_data_path: str = "data/pdf"
    
    # CORS Settings
    cors_origins: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # Research Settings
    default_research_depth: str = "standard"
    max_research_iterations: int = 5
    max_research_breadth: int = 10
    
    @property
    def model_name(self) -> str:
        """Get the model name based on provider."""
        if self.llm_provider == "google":
            return self.google_model
        elif self.llm_provider == "megallm":
            return self.megallm_model
        return self.openai_model
    
    @property
    def effective_google_api_key(self) -> Optional[str]:
        """Get Google API key from either google_api_key or gemini_api_key."""
        return self.google_api_key or self.gemini_api_key
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = Settings()
