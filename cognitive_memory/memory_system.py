"""Optimized cognitive memory system implementation."""

import os
import time
from collections import deque
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from strands.models import BedrockModel
from strands import Agent
from .models import MemoryItem, CognitiveState
from .vector_store import VectorStore
from .config import get_logger, load_config

logger = get_logger("core.memory_system")
config = load_config()


@dataclass
class MemoryStatus:
    """Memory system status snapshot."""
    working_items: int
    episodic_items: int
    vector_items: int
    confidence: float
    gaps: List[str]


class CognitiveMemorySystem:
    """Cognitive memory system with working memory, episodic storage, and vector persistence."""
    
    def __init__(self, 
                 embedding_model_id: str = "amazon.titan-embed-text-v1", 
                 synthesis_model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"):
        """Initialize cognitive memory system with embedding and synthesis models."""
        self.embedding_model = BedrockModel(model_id=embedding_model_id, max_tokens=config.model.max_tokens)
        self.synthesis_model = BedrockModel(model_id=synthesis_model_id, max_tokens=config.model.max_tokens)
        self._synthesis_agent = Agent(model=self.synthesis_model)
        
        self.working_buffer = deque(maxlen=64)
        self.episodic_buffer = deque(maxlen=256)
        
        self.vector_store = VectorStore(
            embedding_model=self.embedding_model,
            chroma_host=os.getenv("CHROMA_HOST", "localhost"),
            chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
            collection_name=os.getenv("CHROMA_COLLECTION", "cognitive_memory")
        )
        
        self.attention_threshold = 0.7
        self.consolidation_threshold = 0.3
        self.current_time = 0
        self.operation_logs = []
        self.cognitive_state = None

    def process_task(self, task: str, documents: List[str] = None) -> Dict[str, Any]:
        """Process task using memory reuse and document analysis. Returns result dictionary for compatibility."""
        if existing := self._check_task_reuse(task):
            return {
                'final_synthesis': existing,
                'metacognitive_status': {'confidence_score': 0.9}
            }
            
        if not documents:
            return {
                'final_synthesis': "No documents provided for processing.",
                'metacognitive_status': {'confidence_score': 0.1}
            }
        
        self._index_documents(task, documents)
        subtasks = self._decompose_task(task)
        insights = [self._process_subtask(subtask) for subtask in subtasks]
        
        synthesis = self._synthesize(insights, f"Task: {task}")
        status = self.get_memory_status()

        return {
            'final_synthesis': synthesis,
            'metacognitive_status': {
                'confidence_score': status.confidence,
                'information_gaps': status.gaps
            }
        }

    def get_memory_status(self) -> MemoryStatus:
        """Get current memory system status including buffer sizes and confidence."""
        return MemoryStatus(
            working_items=len(self.working_buffer),
            episodic_items=len(self.episodic_buffer),
            vector_items=self.vector_store.count(),
            confidence=self._calculate_confidence(),
            gaps=self._detect_gaps()
        )

    def _check_task_reuse(self, task: str) -> Optional[str]:
        """Check memory buffers first, then vector store for task reuse."""
        if memory_items := self._search_memory_buffers(task):
            self.operation_logs.append({"type": "task_memory_reuse", "items": len(memory_items)})
            return self._synthesize([item.content for item in memory_items], task)
        
        if memory_items := self.vector_store.search(task, top_k=3):
            self.operation_logs.append({"type": "task_vector_reuse", "items": len(memory_items)})
            content = []
            for item in memory_items:
                if item[1] > 0.85:
                    content.append(item[2])
                    self._add_to_working_memory(item[2], task)
            return self._synthesize(content, task)
        return None

    def _index_documents(self, task: str, documents: List[str]) -> None:
        """Chunk and index documents into vector store for retrieval."""
        for doc in documents:
            chunks = self._chunk_document(doc)
            for chunk in chunks:
                if len(chunk.strip()) > 50:
                    self.vector_store.add(chunk, {"task": task, "source": "document"})

    def _chunk_document(self, document: str) -> List[str]:
        """Split document into semantic chunks respecting sentence boundaries."""
        sentences = [s.strip() + "." for s in document.split(".") if s.strip()]
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < 300:
                current_chunk += " " + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def _decompose_task(self, task: str) -> List[str]:
        """Break down complex task into manageable subtasks using LLM."""
        try:
            response = str(self._synthesis_agent(f"Break down this task into subtasks. You must not create more than 3 categories with no more than 2 items: {task}"))
            subtasks = [line.strip().lstrip("0123456789.-) ") 
                       for line in response.split("\n") 
                       if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith("-"))]
            return subtasks if subtasks else ["Analyze", "Process", "Synthesize"]
        except Exception:
            return ["Analyze", "Process", "Synthesize"]

    def _process_subtask(self, subtask: str) -> str:
        """Process individual subtask using memory reuse or vector retrieval."""
        return self._check_task_reuse(subtask)

    def _search_memory_buffers(self, query: str) -> List[MemoryItem]:
        """Search working and episodic memory buffers for relevant items using cosine similarity."""
        self._consolidate_buffers()

        if not self.working_buffer and not self.episodic_buffer:
            return []

        query_embedding = self.vector_store.embed(query)
        relevant_items = []
        all_items = list(self.working_buffer) + list(self.episodic_buffer)
        
        for item in all_items:
            if hasattr(item, 'embedding') and len(item.embedding) > 0:
                similarity = cosine_similarity([query_embedding], [item.embedding])[0][0]
                
                if similarity > self.attention_threshold and len(item.content) > 50:
                    item.boost()
                    relevant_items.append(item)
        
        return relevant_items

    def _add_to_working_memory(self, content: str, source: str) -> None:
        """Add new content to working memory, avoiding duplicates."""
        for item in self.working_buffer:
            if item.content == content:
                item.boost()
                return
        
        memory_item = MemoryItem(
            content=content,
            embedding=self.vector_store.embed(content),
            creation_time=self.current_time,
            last_access_time=self.current_time,
            task_context=source,
            source="generated"
        )
        
        self.working_buffer.append(memory_item)

    def _consolidate_buffers(self) -> None:
        """Apply forgetting curve, remove low-relevance items, promote important memories."""
        self.current_time += 1
        
        for item in list(self.working_buffer) + list(self.episodic_buffer):
            item.decay(self.current_time)
        
        self.working_buffer = deque(
            [item for item in self.working_buffer if item.relevance_score > 0.3],
            maxlen=self.working_buffer.maxlen
        )
        
        for item in self.working_buffer:
            if (item.access_count > 2 and 
                not any(id(item) == id(e) for e in self.episodic_buffer)):
                self.episodic_buffer.append(item)

    def _synthesize(self, contents: List[str], context: str) -> str:
        """Generate coherent synthesis from multiple content pieces using LLM."""
        if not contents:
            return ""
        
        combined = "\n\n".join(contents[:5])
        
        prompt = f"""Context: {context}
        
Information:
{combined}

Provide a concise, informative synthesis:"""
        
        try:
            return str(self._synthesis_agent(prompt))
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return f"Analysis of: {combined[:200]}..."

    def _calculate_confidence(self) -> float:
        """Calculate confidence score based on memory relevance and utilization."""
        if not self.working_buffer:
            return 0.0
        
        avg_relevance = np.mean([item.relevance_score for item in self.working_buffer])
        buffer_utilization = len(self.working_buffer) / self.working_buffer.maxlen
        
        return min(1.0, (avg_relevance * 0.7 + buffer_utilization * 0.3))

    def _detect_gaps(self) -> List[str]:
        """Identify information gaps in current memory state."""
        gaps = []
        
        if len(self.working_buffer) < 3:
            gaps.append("Limited working memory")
        
        if self.vector_store.count() < 5:
            gaps.append("Insufficient knowledge base")
        
        return gaps

    def get_metacognitive_status(self) -> Dict[str, Any]:
        """Legacy method returning memory status as dictionary for backward compatibility."""
        status = self.get_memory_status()
        return {
            "confidence_score": status.confidence,
            "information_gaps": status.gaps,
            "memory_utilization": {
                "working_buffer": status.working_items,
                "episodic_buffer": status.episodic_items,
                "vector_store": status.vector_items
            }
        }
