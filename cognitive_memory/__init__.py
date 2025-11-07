"""
Cognitive Memory System - Advanced memory capabilities for AI agents.

This package provides a production-ready cognitive memory system that implements
human-like learning patterns including semantic clustering, forgetting curve decay,
memory consolidation, and progressive reasoning.
"""

from .memory_system import CognitiveMemorySystem
from .models import MemoryItem, CognitiveState
from .analyzer import MemoryAnalyzer
from .vector_store import VectorStore

__version__ = "0.1.0"
__author__ = "Cognitive Memory Contributors"
__all__ = [
    "CognitiveMemorySystem",
    "MemoryItem", 
    "CognitiveState",
    "MemoryAnalyzer",
    "VectorStore",
]
