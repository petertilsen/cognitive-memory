"""Example usage of the cognitive memory system."""

from cognitive_memory import CognitiveMemorySystem, MemoryAnalyzer


def main():
    """Example: Building a research assistant with cognitive memory."""
    
    # Step 1: Initialize the memory system
    memory_system = CognitiveMemorySystem()
    
    # Step 2: (Optional) Add analyzer for monitoring
    analyzer = MemoryAnalyzer(memory_system)
    
    # Step 3: Use the system to process research queries with documents
    
    # First research session - AI basics
    ai_documents = [
        "Artificial Intelligence (AI) refers to computer systems that can perform tasks typically requiring human intelligence, such as visual perception, speech recognition, and decision-making.",
        "Machine learning is a subset of AI that enables systems to automatically learn and improve from experience without being explicitly programmed for every scenario."
    ]
    
    result1 = memory_system.process_task("What is artificial intelligence?", ai_documents)
    print("Query 1 Response:", result1['final_synthesis'][:100] + "...")
    
    # Second research session - builds on previous knowledge
    ml_documents = [
        "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data, enabling breakthroughs in image recognition and natural language processing.",
        "Supervised learning trains models on labeled data, while unsupervised learning finds patterns in unlabeled data."
    ]
    
    result2 = memory_system.process_task("How does machine learning work?", ml_documents)
    print("Query 2 Response:", result2['final_synthesis'][:100] + "...")
    
    # Third query - should reuse previous knowledge
    result3 = memory_system.process_task("What's the difference between AI and machine learning?")
    print("Query 3 Response:", result3['final_synthesis'][:100] + "...")
    
    # Step 4: Check what the system learned
    print(f"\nMemory Status:")
    print(f"- Working Memory: {analyzer.get_memory_utilization()['working_buffer']['size']} items")
    print(f"- Reuse Rate: {analyzer.get_reuse_stats()['reuse_rate']:.1%}")
    print(f"- Total Events: {len(analyzer.events)}")


if __name__ == "__main__":
    main()
