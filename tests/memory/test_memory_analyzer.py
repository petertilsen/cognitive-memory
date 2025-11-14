"""Unit tests for optimized memory analyzer."""

import unittest
from unittest.mock import Mock
from collections import defaultdict, deque

from cognitive_memory.analyzer import MemoryAnalyzer
from cognitive_memory.memory_system import MemoryStatus


class TestMemoryAnalyzer(unittest.TestCase):
    """Test cases for MemoryAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_memory_system = Mock()
        self.mock_memory_system.working_buffer = Mock()
        self.mock_memory_system.working_buffer.maxlen = 64
        self.mock_memory_system.episodic_buffer = Mock()
        self.mock_memory_system.episodic_buffer.maxlen = 256
        
        # Mock get_memory_status
        self.mock_memory_system.get_memory_status.return_value = MemoryStatus(
            working_items=5,
            episodic_items=10,
            vector_items=20,
            confidence=0.8,
            gaps=[]
        )
        
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = MemoryAnalyzer(self.mock_memory_system)
        
        self.assertEqual(analyzer.memory_system, self.mock_memory_system)
        self.assertIsInstance(analyzer.events, deque)
        self.assertIsInstance(analyzer.stats, defaultdict)
        self.assertEqual(analyzer.memory_system._event_handler, analyzer._handle_event)
        
    def test_handle_event(self):
        """Test event handling."""
        analyzer = MemoryAnalyzer(self.mock_memory_system)
        
        analyzer._handle_event("memory_reuse", {"type": "buffer", "items": 3})
        
        self.assertEqual(len(analyzer.events), 1)
        self.assertEqual(analyzer.stats["memory_reuse"], 1)
        self.assertEqual(analyzer.stats["reuse_buffer"], 3)
        
    def test_get_memory_utilization(self):
        """Test memory utilization metrics."""
        analyzer = MemoryAnalyzer(self.mock_memory_system)
        utilization = analyzer.get_memory_utilization()
        
        self.assertIn("working_buffer", utilization)
        self.assertIn("episodic_buffer", utilization)
        self.assertIn("vector_store", utilization)
        self.assertIn("confidence", utilization)
        
        self.assertEqual(utilization["working_buffer"]["size"], 5)
        self.assertEqual(utilization["working_buffer"]["capacity"], 64)
        self.assertAlmostEqual(utilization["working_buffer"]["utilization"], 5/64)
        
    def test_get_reuse_stats(self):
        """Test reuse statistics."""
        analyzer = MemoryAnalyzer(self.mock_memory_system)
        
        # Add some events
        analyzer._handle_event("memory_reuse", {"type": "buffer", "items": 2})
        analyzer._handle_event("memory_reuse", {"type": "vector", "items": 3})
        analyzer._handle_event("document_indexing", {"documents": 1})
        
        stats = analyzer.get_reuse_stats()
        
        self.assertEqual(stats["total_reuse_events"], 2)
        self.assertEqual(stats["buffer_reuse_items"], 2)
        self.assertEqual(stats["vector_reuse_items"], 3)
        self.assertAlmostEqual(stats["reuse_rate"], 2/3)  # 2 reuse events out of 3 total
        
    def test_generate_memory_report(self):
        """Test memory report generation."""
        analyzer = MemoryAnalyzer(self.mock_memory_system)
        
        # Add some events
        analyzer._handle_event("memory_reuse", {"type": "buffer", "items": 1})
        
        report = analyzer.generate_memory_report()
        
        self.assertIn("reuse_analysis", report)
        self.assertIn("buffer_analysis", report)
        self.assertIn("reuse_rate", report["reuse_analysis"])
        self.assertIn("working_buffer", report["buffer_analysis"])
        self.assertIn("episodic_buffer", report["buffer_analysis"])
        self.assertIn("vector_store", report["buffer_analysis"])
        
    def test_compare_memory_states(self):
        """Test memory state comparison."""
        analyzer = MemoryAnalyzer(self.mock_memory_system)
        
        before = {
            "buffer_analysis": {
                "working_buffer": {"size": 3},
                "episodic_buffer": {"size": 8}
            }
        }
        
        after = {
            "buffer_analysis": {
                "working_buffer": {"size": 5},
                "episodic_buffer": {"size": 10}
            }
        }
        
        comparison = analyzer.compare_memory_states(before, after)
        
        self.assertIn("buffer_changes", comparison)
        self.assertEqual(comparison["buffer_changes"]["working_buffer"]["change"], 2)
        
    def test_empty_stats(self):
        """Test analyzer with no events."""
        analyzer = MemoryAnalyzer(self.mock_memory_system)
        
        stats = analyzer.get_reuse_stats()
        
        self.assertEqual(stats["total_reuse_events"], 0)
        self.assertEqual(stats["buffer_reuse_items"], 0)
        self.assertEqual(stats["vector_reuse_items"], 0)
        self.assertEqual(stats["reuse_rate"], 0)


if __name__ == '__main__':
    unittest.main()
