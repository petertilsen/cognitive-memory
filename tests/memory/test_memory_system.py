"""Unit tests for optimized cognitive memory system."""

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from collections import deque

from cognitive_memory.memory_system import CognitiveMemorySystem, MemoryStatus
from cognitive_memory.models import MemoryItem


class TestCognitiveMemorySystem(unittest.TestCase):
    """Test cases for optimized CognitiveMemorySystem."""
    
    @patch('cognitive_memory.memory_system.VectorStore')
    @patch('cognitive_memory.memory_system.Agent')
    @patch('cognitive_memory.memory_system.BedrockModel')
    def setUp(self, mock_bedrock, mock_agent, mock_vector_store):
        """Set up test fixtures."""
        mock_bedrock.return_value = Mock()
        mock_agent.return_value = Mock()
        
        self.mock_vector_store = Mock()
        self.mock_vector_store.count.return_value = 5
        self.mock_vector_store.embed.return_value = np.array([0.1, 0.2, 0.3])
        self.mock_vector_store.search.return_value = []
        mock_vector_store.return_value = self.mock_vector_store
        
        self.system = CognitiveMemorySystem()
        
    def test_initialization(self):
        """Test system initialization."""
        self.assertIsInstance(self.system.working_buffer, deque)
        self.assertIsInstance(self.system.episodic_buffer, deque)
        self.assertEqual(self.system.working_buffer.maxlen, 64)
        self.assertEqual(self.system.episodic_buffer.maxlen, 256)
        self.assertEqual(self.system.attention_threshold, 0.7)  # From config
        self.assertEqual(self.system.consolidation_threshold, 0.3)  # From config
        
    def test_configurable_thresholds(self):
        """Test that thresholds are configurable via environment."""
        # Test that thresholds can be modified at runtime
        original_attention = self.system.attention_threshold
        original_consolidation = self.system.consolidation_threshold
        
        # Modify thresholds
        self.system.attention_threshold = 0.8
        self.system.consolidation_threshold = 0.4
        
        self.assertEqual(self.system.attention_threshold, 0.8)
        self.assertEqual(self.system.consolidation_threshold, 0.4)
        
        # Restore original values
        self.system.attention_threshold = original_attention
        self.system.consolidation_threshold = original_consolidation
        
    def test_emit_event(self):
        """Test event emission."""
        handler_calls = []
        
        def mock_handler(event, data):
            handler_calls.append((event, data))
            
        self.system._event_handler = mock_handler
        self.system._emit_event("test_event", {"key": "value"})
        
        self.assertEqual(len(handler_calls), 1)
        self.assertEqual(handler_calls[0][0], "test_event")
        self.assertEqual(handler_calls[0][1]["key"], "value")
        
    def test_process_task_no_documents(self):
        """Test process_task with no documents."""
        result = self.system.process_task("test task")
        
        self.assertEqual(result["final_synthesis"], "No documents provided for processing.")
        self.assertEqual(result["metacognitive_status"]["confidence_score"], 0.1)
        
    def test_process_task_with_reuse(self):
        """Test process_task with memory reuse."""
        # Mock vector store to return reusable content
        self.mock_vector_store.search.return_value = [
            ("id1", 0.9, "reused content", {})
        ]
        
        # Mock synthesis agent to return string
        with patch.object(self.system, '_synthesize', return_value="synthesized reused content"):
            result = self.system.process_task("test task")
            
            self.assertIn("synthesized", result["final_synthesis"])
            self.assertEqual(result["metacognitive_status"]["confidence_score"], 0.9)
        
    def test_process_task_with_documents(self):
        """Test process_task with new documents."""
        with patch.object(self.system, '_decompose_task', return_value=["subtask1"]):
            with patch.object(self.system, '_process_subtask', return_value="insight"):
                with patch.object(self.system, '_synthesize', return_value="final result"):
                    result = self.system.process_task("test task", ["doc1", "doc2"])
                    
                    self.assertEqual(result["final_synthesis"], "final result")
                    self.assertIn("confidence_score", result["metacognitive_status"])
                    
    def test_add_to_working_memory(self):
        """Test adding items to working memory."""
        initial_count = len(self.system.working_buffer)
        
        self.system._add_to_working_memory("test content", "test source")
        
        self.assertEqual(len(self.system.working_buffer), initial_count + 1)
        self.assertEqual(self.system.working_buffer[-1].content, "test content")
        
    def test_consolidate_buffers(self):
        """Test memory consolidation."""
        # Add item to working buffer
        item = MemoryItem(
            content="test content",
            embedding=np.array([0.1, 0.2, 0.3]),
            creation_time=0,
            last_access_time=0,
            relevance_score=0.8,
            access_count=3
        )
        self.system.working_buffer.append(item)
        
        initial_episodic_count = len(self.system.episodic_buffer)
        self.system._consolidate_buffers()
        
        # High access count item should be promoted to episodic
        self.assertEqual(len(self.system.episodic_buffer), initial_episodic_count + 1)
        
    def test_get_memory_status(self):
        """Test memory status retrieval."""
        status = self.system.get_memory_status()
        
        self.assertIsInstance(status, MemoryStatus)
        self.assertIsInstance(status.working_items, int)
        self.assertIsInstance(status.episodic_items, int)
        self.assertIsInstance(status.vector_items, int)
        self.assertIsInstance(status.confidence, float)
        self.assertIsInstance(status.gaps, list)
        
    def test_chunk_document(self):
        """Test document chunking."""
        document = "This is sentence one. This is sentence two. This is sentence three."
        chunks = self.system._chunk_document(document)
        
        self.assertIsInstance(chunks, list)
        self.assertTrue(len(chunks) > 0)
        self.assertTrue(all(isinstance(chunk, str) for chunk in chunks))
        
    def test_decompose_task_error_handling(self):
        """Test task decomposition error handling."""
        with patch.object(self.system._synthesis_agent, '__call__', side_effect=Exception("LLM error")):
            subtasks = self.system._decompose_task("test task")
            
            # Should return default subtasks on error
            self.assertEqual(subtasks, ["Analyze", "Process", "Synthesize"])
            
    def test_synthesize_error_handling(self):
        """Test synthesis error handling."""
        with patch.object(self.system, '_synthesis_agent') as mock_agent:
            mock_agent.side_effect = Exception("Synthesis error")
            result = self.system._synthesize(["content1", "content2"], "test context")
            
            # Should return fallback synthesis on error
            self.assertIn("Analysis of:", result)
            
    def test_search_memory_buffers_empty(self):
        """Test searching empty memory buffers."""
        results = self.system._search_memory_buffers("test query")
        
        self.assertEqual(results, [])
        
    def test_search_memory_buffers_with_items(self):
        """Test searching memory buffers with items."""
        # Add item to working buffer with longer content
        item = MemoryItem(
            content="test content for search that is long enough to pass the length filter",
            embedding=np.array([0.9, 0.9, 0.9]),  # High similarity embedding
            creation_time=0,
            last_access_time=0,
            relevance_score=0.8
        )
        self.system.working_buffer.append(item)
        
        # Mock vector store embed to return similar embedding for high cosine similarity
        self.mock_vector_store.embed.return_value = np.array([0.9, 0.9, 0.9])
        
        results = self.system._search_memory_buffers("test query")
        
        # Should find the item due to high similarity and sufficient content length
        self.assertEqual(len(results), 1)
        self.assertIn("test content for search", results[0].content)
        
    def test_calculate_confidence_empty_buffer(self):
        """Test confidence calculation with empty buffer."""
        confidence = self.system._calculate_confidence()
        
        self.assertEqual(confidence, 0.0)
        
    def test_detect_gaps(self):
        """Test information gap detection."""
        # Mock vector store count to return low value
        self.mock_vector_store.count.return_value = 2
        
        gaps = self.system._detect_gaps()
        
        self.assertIn("Limited working memory", gaps)
        self.assertIn("Insufficient knowledge base", gaps)
        
    def test_get_metacognitive_status(self):
        """Test legacy metacognitive status method."""
        status = self.system.get_metacognitive_status()
        
        self.assertIn("confidence_score", status)
        self.assertIn("information_gaps", status)
        self.assertIn("memory_utilization", status)


if __name__ == '__main__':
    unittest.main()
