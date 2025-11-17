"""Configuration and logging utilities for cognitive memory system."""

import logging
import os
from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Model configuration settings."""
    max_tokens: int
    temperature: float

@dataclass
class VectorStoreConfig:
    """Vector store configuration."""
    chroma_host: str
    chroma_port: int
    collection_name: str


@dataclass
class MemoryConfig:
    """Memory system configuration."""
    working_buffer_size: int
    episodic_buffer_size: int
    attention_threshold: float
    consolidation_threshold: float
    decay_rate: float

@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig
    memory: MemoryConfig
    vector_store: VectorStoreConfig
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    
    return logger


def load_config() -> Config:
    """Load configuration from environment or defaults."""
    return Config(
        model=ModelConfig(
            max_tokens=int(os.getenv("MAX_TOKENS", "1000")),
            temperature=float(os.getenv("TEMPERATURE", "0.7"))
        ),
        memory=MemoryConfig(
            working_buffer_size=int(os.getenv("WORKING_BUFFER_SIZE", "64")),
            episodic_buffer_size=int(os.getenv("EPISODIC_BUFFER_SIZE", "256")),
            attention_threshold=float(os.getenv("ATTENTION_THRESHOLD", "0.7")),
            consolidation_threshold=float(os.getenv("CONSOLIDATION_THRESHOLD", "0.3")),
            decay_rate=float(os.getenv("DECAY_RATE", "0.1"))
        ),
        vector_store=VectorStoreConfig(
            chroma_host=os.getenv("CHROMA_HOST", "localhost"),
            chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
            collection_name=os.getenv("COLLECTION_NAME", "cognitive_memory")
        )
    )
