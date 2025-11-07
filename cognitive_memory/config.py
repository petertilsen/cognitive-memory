"""Configuration and logging utilities for cognitive memory system."""

import logging
import os
from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Model configuration settings."""
    max_tokens: int = 1000
    temperature: float = 0.7


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)


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
        logger.setLevel(logging.INFO)
    
    return logger


def load_config() -> Config:
    """Load configuration from environment or defaults."""
    return Config(
        model=ModelConfig(
            max_tokens=int(os.getenv("MAX_TOKENS", "1000")),
            temperature=float(os.getenv("TEMPERATURE", "0.7"))
        )
    )
