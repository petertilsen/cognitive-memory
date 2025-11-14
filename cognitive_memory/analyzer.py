"""Memory system analyzer using observer pattern."""

from typing import Dict, Any, List
from collections import defaultdict, deque
from .memory_system import CognitiveMemorySystem


class MemoryAnalyzer:
    """Memory system analyzer for cognitive memory analysis and reporting."""
    
    def __init__(self, memory_system: CognitiveMemorySystem):
        """Initialize analyzer with memory system reference."""
        self.memory_system = memory_system
        self.events = deque(maxlen=1000)
        self.stats = defaultdict(int)
        
        # Attach as event handler
        self.memory_system._event_handler = self._handle_event
    
    def _handle_event(self, event: str, data: Dict[str, Any]) -> None:
        """Handle memory system events."""
        self.events.append({"event": event, "data": data})
        self.stats[event] += 1
        if event == "memory_reuse":
            self.stats[f"reuse_{data['type']}"] += data.get('items', 0)
    
    def get_memory_utilization(self) -> Dict[str, Any]:
        """Get current memory utilization metrics."""
        status = self.memory_system.get_memory_status()
        return {
            "working_buffer": {
                "size": status.working_items,
                "capacity": self.memory_system.working_buffer.maxlen,
                "utilization": status.working_items / self.memory_system.working_buffer.maxlen
            },
            "episodic_buffer": {
                "size": status.episodic_items,
                "capacity": self.memory_system.episodic_buffer.maxlen,
                "utilization": status.episodic_items / self.memory_system.episodic_buffer.maxlen
            },
            "vector_store": {
                "size": status.vector_items
            },
            "confidence": status.confidence
        }
    
    def get_reuse_stats(self) -> Dict[str, Any]:
        """Get memory reuse statistics."""
        total_reuse = self.stats.get("memory_reuse", 0)
        buffer_reuse = self.stats.get("reuse_buffer", 0)
        vector_reuse = self.stats.get("reuse_vector", 0)
        
        return {
            "total_reuse_events": total_reuse,
            "buffer_reuse_items": buffer_reuse,
            "vector_reuse_items": vector_reuse,
            "reuse_rate": total_reuse / max(1, len(self.events)) if self.events else 0
        }
    
    def generate_memory_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory report for demo compatibility."""
        utilization = self.get_memory_utilization()
        reuse_stats = self.get_reuse_stats()
        
        return {
            "reuse_analysis": {
                "reuse_rate": reuse_stats["reuse_rate"]
            },
            "buffer_analysis": {
                "working_buffer": utilization["working_buffer"],
                "episodic_buffer": utilization["episodic_buffer"],
                "vector_store": utilization["vector_store"]
            }
        }
    
    def compare_memory_states(self, before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two memory states for demo compatibility."""
        return {
            "buffer_changes": {
                "working_buffer": {
                    "change": after["buffer_analysis"]["working_buffer"]["size"] - 
                             before["buffer_analysis"]["working_buffer"]["size"]
                }
            }
        }
