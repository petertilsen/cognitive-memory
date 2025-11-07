"""Example usage of the cognitive memory system."""

import os
from cognitive_memory import CognitiveMemorySystem, MemoryAnalyzer


def main():
    """Demonstrate cognitive memory system usage."""
    
    # Initialize cognitive memory system
    memory = CognitiveMemorySystem(
        embedding_model_id="amazon.titan-embed-text-v1",
        synthesis_model_id="anthropic.claude-3-haiku-20240307-v1:0"
    )
    
    # Example queries showing progressive learning
    queries = [
        "What are neural networks?",
        "What is backpropagation?", 
        "How do neural networks learn?",
        "What are the applications of deep learning?"
    ]
    
    print("🧠 Cognitive Memory System Demo")
    print("=" * 50)
    
    for i, query in enumerate(queries, 1):
        print(f"\n--- Query {i} ---")
        print(f"Question: {query}")
        
        # Process query with cognitive memory
        result = memory.process_task(query, [])
        
        print(f"Response: {result['final_synthesis'][:200]}...")
        
        # Show memory analytics
        analyzer = MemoryAnalyzer(memory)
        report = analyzer.generate_memory_report()
        
        reuse_rate = report['reuse_analysis']['reuse_rate']
        working_size = report['buffer_analysis']['working_buffer']['size']
        
        print(f"Memory Reuse: {reuse_rate:.1%}")
        print(f"Working Memory: {working_size} items")


if __name__ == "__main__":
    main()
