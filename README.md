# RAG Evaluation Framework

This repository contains a comprehensive evaluation framework for testing and optimizing Retrieval Augmented Generation (RAG) systems. It allows systematic comparison of different retrieval approaches, chunking strategies, and configuration parameters.

## 📋 Overview

RAG systems combine retrieval of relevant information with text generation to produce more accurate and grounded responses. This framework helps evaluate and optimize RAG performance by:

1. Testing different retrieval approaches (Semantic Search, RAG Fusion, Hybrid Search, HyDE)
2. Comparing various text chunking strategies (Character, Recursive, Semantic, SDPM, Agentic)
3. Evaluating different chunk sizes and configurations
4. Measuring performance across multiple metrics (faithfulness, precision, recall, relevancy)
5. Tracking timing and efficiency metrics

## 🧠 Key Components

- **Retrieval Approaches**:
  - **Semantic Search**: Uses embeddings to find semantically similar content
  - **RAG Fusion**: Combines multiple retrieval results for improved coverage
  - **Hybrid Search**: Combines semantic search with keyword-based search
  - **HyDE**: Hypothetical Document Embeddings for improved retrieval

- **Chunking Strategies**:
  - **Character**: Simple fixed-size text splitting
  - **Recursive**: Hierarchical chunking using various separators
  - **Semantic**: Chunks based on semantic coherence
  - **SDPM**: Semantic Double Pass Merging for improved coherence
  - **Agentic**: LLM-assisted chunking that finds natural breakpoints

## 📊 Metrics

### Quality Metrics
- **Faithfulness**: How well responses are grounded in the retrieved context
- **Answer Relevancy**: How relevant responses are to the questions
- **Context Precision**: How much of the retrieved context is relevant
- **Context Recall**: How much relevant information was retrieved

### Performance Metrics
- **Timing Metrics**: Total time, chunking time, ingestion time, query processing time, evaluation time
- **Volume Metrics**: Document count, total chunks, average chunks per document

### Running Evaluations
The main script supports different evaluation modes:

```bash
nohup python main.py > output.log &
```

## 📝 Documentation

- **tests.md**: Documents test configurations and results
- **recap.md**: Provides a summary of work done and findings
- **visualization_comparison.py**: Script for visualizing results

## 📊 Visualization

The framework includes comprehensive visualization capabilities to analyze results:

- Radar charts comparing retrieval approaches
- Bar charts showing metric performance
- Heatmaps visualizing performance across strategies
- Precision-recall curves
- Timing and performance comparisons

To generate visualizations:
```bash
python visualization_comparison.py
```

## 📈 Key Findings So Far

1. Semantic Search consistently outperforms other retrieval approaches in terms of faithfulness and precision
2. SDPM and Semantic chunking strategies generally yield the best results
3. There's a clear trade-off between precision and recall, with Semantic Search offering the best balance
4. Chunk size significantly impacts both quality and performance metrics

## 🙏 Acknowledgements

- RAGAS for evaluation metrics
- LangChain for chunking utilities
- R2R framework for RAG implementation 