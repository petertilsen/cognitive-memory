# 🧠 Cognitive Memory

![Tests](https://github.com/petertilsen/cognitive-memory/workflows/Tests/badge.svg)
![Coverage](https://codecov.io/gh/petertilsen/cognitive-memory/branch/main/graph/badge.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)

A production-ready cognitive memory system for AI agents with human-like learning and memory patterns.

## ✅ Implemented Features

### 🧠 Two-Layer Memory Architecture
- **Working Buffer (64 items)**: Active processing and short-term memory
- **Episodic Buffer (256 items)**: Recent experiences and learned patterns  
- **Vector Store (∞)**: Persistent long-term knowledge with ChromaDB

### 📉 Forgetting Curve Implementation
Implements Ebbinghaus forgetting curve for natural memory decay:
- **Exponential Decay**: `relevance = relevance * exp(-decay_rate * time_diff)`
- **Access-based Reinforcement**: Frequently accessed memories resist decay
- **Overflow Protection**: Prevents mathematical overflow in extreme cases

### 🔄 Memory Consolidation
Intelligent promotion system based on access patterns:
- **Automatic Promotion**: High-access items (>2 accesses) move to episodic buffer
- **Relevance Filtering**: Low-relevance items (<0.3) are removed during consolidation
- **Time-based Processing**: Consolidation occurs during memory searches

### 🎯 Attention Filtering  
Context-aware memory retrieval with similarity thresholds:
- **Cosine Similarity**: Vector-based relevance scoring
- **Attention Threshold**: 0.7 similarity minimum for retrieval
- **Content Length Filter**: Minimum 50 characters for meaningful memories

### ⚡ Performance Optimizations
- **Memory-first Reuse**: Checks buffers before expensive vector searches
- **Vectorized Similarity**: NumPy-based batch cosine similarity calculations
- **Efficient Chunking**: 300-character chunks respecting sentence boundaries
- **Event-driven Analytics**: Observer pattern for zero-overhead monitoring

### 📊 Memory Analytics
Real-time monitoring and reporting:
- **Memory Utilization**: Buffer capacity and usage tracking
- **Reuse Statistics**: Memory vs vector reuse rates and patterns
- **Event Tracking**: Document indexing, consolidation, and reuse events
- **State Comparison**: Before/after memory state analysis

## ❌ Missing Human-like Features

### 🔗 Semantic Clustering
- **Status**: Removed (was computed but unused)
- **Gap**: No automatic organization by conceptual similarity
- **Impact**: Memories not grouped by related concepts

### 🤔 Advanced Metacognition
- **Status**: Basic confidence scoring only
- **Gap**: No proactive information need identification
- **Impact**: Limited self-awareness of knowledge gaps

### 🧩 Cross-domain Learning
- **Status**: Not implemented
- **Gap**: No transfer learning between different knowledge domains
- **Impact**: Each topic area learned in isolation

### 🔄 Memory Reconstruction
- **Status**: Not implemented  
- **Gap**: Cannot reconstruct partial memories from fragments
- **Impact**: All-or-nothing memory retrieval

### 📚 Episodic vs Semantic Distinction
- **Status**: Single memory type
- **Gap**: No separation of experiences vs facts
- **Impact**: Cannot distinguish between "what happened" vs "what is true"

## 📦 Installation

```bash
pip install cognitive-memory
```

## 🚀 Quick Start

```python
from cognitive_memory import CognitiveMemorySystem, MemoryAnalyzer

# Initialize system
memory = CognitiveMemorySystem()
analyzer = MemoryAnalyzer(memory)  # Optional: for monitoring

# Process tasks with documents
result = memory.process_task(
    "What is machine learning?", 
    ["Machine learning is a subset of AI that enables systems to learn from data..."]
)

print(result['final_synthesis'])

# Check memory utilization
utilization = analyzer.get_memory_utilization()
print(f"Working memory: {utilization['working_buffer']['utilization']:.1%}")
```

## 📊 Memory Reuse Performance

```python
# Progressive learning demonstration
queries = [
    "What is AI?",           # 0% reuse (new topic)
    "How does AI work?",     # 15-25% reuse (builds on AI knowledge)  
    "AI applications?",      # 25-40% reuse (leverages accumulated AI understanding)
]

for query in queries:
    result = memory.process_task(query, documents)
    stats = analyzer.get_reuse_stats()
    print(f"Query: {query}")
    print(f"Reuse rate: {stats['reuse_rate']:.1%}")
```

## 🛠️ Requirements

### Prerequisites
- **Python 3.8+**
- **AWS Bedrock Access**: For embeddings (`amazon.titan-embed-text-v1`) and synthesis models
- **ChromaDB Server**: For persistent vector storage

### ChromaDB Setup
```bash
# Install and run ChromaDB
pip install chromadb
chroma run --host localhost --port 8000
```

### Environment Variables
```env
# AWS Bedrock (required)
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key  
AWS_REGION=us-east-1

# ChromaDB (optional, defaults shown)
CHROMA_HOST=localhost
CHROMA_PORT=8000
CHROMA_COLLECTION=cognitive_memory

# Memory Configuration (optional, defaults shown)
ATTENTION_THRESHOLD=0.7        # Minimum similarity for memory retrieval
CONSOLIDATION_THRESHOLD=0.3    # Minimum relevance to avoid cleanup
```

## 🏗️ Architecture

```
Memory Flow:
Documents → Chunking → Vector Store → Memory Buffers → Synthesis

Core Components:
├── CognitiveMemorySystem    # Main API and memory management
├── MemoryAnalyzer          # Event monitoring and analytics  
├── MemoryItem              # Individual memory with decay/boost
├── VectorStore             # ChromaDB integration
└── MemoryStatus            # Structured status reporting
```

## 📈 Performance Characteristics

- **Memory Reuse**: 15-40% typical reuse rates vs 0% traditional RAG
- **Response Speed**: 25x faster for reused content
- **Memory Efficiency**: Automatic consolidation and cleanup
- **Persistence**: Survives application restarts via ChromaDB

## 🔧 Advanced Usage

### Event Monitoring
```python
def my_event_handler(event, data):
    print(f"Memory event: {event} - {data}")

memory = CognitiveMemorySystem(event_handler=my_event_handler)
```

## 🧪 Testing

```bash
# Run tests with coverage
pytest tests/ --cov=cognitive_memory

# Current coverage: 82% (369 lines, 66 missing)
```

## 📄 License

MIT License - See LICENSE file for details.

---

**Note**: This is an optimized, production-ready implementation focusing on core cognitive memory functionality. Advanced features like semantic clustering and cross-domain learning are identified for future development.
