import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Application Info
    APP_NAME: str = "Voice-Enabled RAG System (Hacker House Goa 2026)"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # LLM Settings (Groq LPU recommended for lowest TTFB)
    GROQ_API_KEY: Optional[str] = Field(default=None, validation_alias="GROQ_API_KEY")
    LLM_MODEL: str = Field(default="openai/gpt-oss-20b", validation_alias="LLM_MODEL")
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 512
    LLM_TIMEOUT_SECONDS: float = 15.0

    # Speech-to-Text Settings (Sarvam AI / Groq Whisper)
    SARVAM_API_KEY: Optional[str] = Field(default=None, validation_alias="SARVAM_API_KEY")
    STT_PROVIDER: str = Field(default="groq_whisper", validation_alias="STT_PROVIDER")
    STT_LANGUAGE_CODE: str = "en-IN"
    
    # Vector Database & Embeddings
    QDRANT_STORAGE_PATH: str = Field(default="./data/qdrant_db", validation_alias="QDRANT_STORAGE_PATH")
    COLLECTION_NAME: str = Field(default="msmarco_xi_hierarchical", validation_alias="COLLECTION_NAME")
    EMBEDDING_MODEL_NAME: str = Field(default="BAAI/bge-small-en-v1.5", validation_alias="EMBEDDING_MODEL_NAME")
    EMBEDDING_DIM: int = 384
    
    # Hierarchical Chunking Parameters
    PARENT_CHUNK_SIZE: int = 400 # tokens/words for parent passage
    PARENT_CHUNK_OVERLAP: int = 50
    CHILD_CHUNK_SIZE: int = 100 # tokens/words for child vector search
    CHILD_CHUNK_OVERLAP: int = 25
    
    # Guardrails & Retrieval Tuning (Calibrated for BGE-small cosine range)
    RELEVANCE_SCORE_THRESHOLD: float = 0.40 # Minimum cosine similarity for child match
    ENABLE_RERANKER: bool = True
    RERANK_SCORE_THRESHOLD: float = 0.50
    TOP_K_CHILDREN: int = 5
    MAX_PARENT_CONTEXTS: int = 3
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000

settings = Settings()
