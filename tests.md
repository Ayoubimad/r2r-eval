# RAG System Testing Documentation

## Overview
This document tracks the testing of different chunking strategies and retrieval approaches for our RAG (Retrieval Augmented Generation) system.

## Retrieval Approaches Tested
1. **Semantic Search**: Standard semantic search using embeddings
2. **RAG Fusion**: Combines multiple retrieval results
3. **Hybrid Search**: Combines semantic search with keyword-based search
4. **HyDE (Hypothetical Document Embeddings)**: Generates hypothetical documents before search

## Chunking Strategies Tested
1. **Character**: Simple character-based chunking with fixed chunk size
2. **Agentic**: LLM-assisted chunking that finds natural breakpoints
3. **Semantic**: Chunks based on semantic coherence
4. **SDPM (Semantic Double Pass Merging)**: Advanced semantic chunking with merging
5. **Recursive**: Hierarchical chunking using different separators

## Key Metrics
- **Faithfulness**: How well responses are grounded in the retrieved context
- **Answer Relevancy**: How relevant responses are to the questions
- **Context Precision**: How much of the retrieved context is relevant
- **Context Recall**: How much relevant information was retrieved

## Performance Metrics
We now track detailed timing and performance statistics for each test:

### Timing Metrics
- **Total Time**: Overall time to complete the full evaluation pipeline
- **Chunking Time**: Time spent dividing documents into chunks
- **Ingestion Time**: Time spent ingesting chunks into the retrieval system
- **Query Processing Time**: Time spent processing RAG queries
- **Evaluation Time**: Time spent evaluating results with RAGAS metrics

### Volume Metrics
- **Document Count**: Number of documents processed
- **Total Chunks**: Total number of chunks created
- **Average Chunks Per Document**: Mean number of chunks per document

These metrics allow us to analyze not just the quality of results but also the computational efficiency of different approaches.

## Test Results Summary

### Initial Testing (Fixed chunk size: 1024 chars)
| Retrieval Approach | Best Strategy | Highest Faithfulness | Best Precision | Best Recall |
|-------------------|---------------|----------------------|----------------|-------------|
| Semantic Search   | Semantic      | 0.9613               | 0.7022         | 0.9682      |
| RAG Fusion        | SDPM          | 0.9249               | 0.3766         | 0.9590      |
| Hybrid Search     | SDPM          | 0.8384               | 0.1742         | 0.9641      |
| HyDE              | Agentic       | 0.8392               | 0.1084         | 0.9793      |

### Key Observations
1. Semantic Search consistently outperforms other retrieval approaches across all metrics except for marginal differences in recall
2. SDPM and Semantic chunking strategies generally yield the best results
3. There's a clear trade-off between precision and recall, with Semantic Search offering the best balance
4. The answer relevancy metric is consistently low across all approaches, suggesting room for improvement

## Chunk Size Testing
Tests with varying chunk sizes to determine optimal chunking parameters for each approach.

| Chunk Size | Status | Notes |
|------------|--------|-------|
| 32         | Pending | Very small chunks, might be inefficient |
| 64         | Pending | |
| 128        | Pending | |
| 256        | Pending | |
| 512        | Pending | |
| 1024       | Completed | Base case, good results with Semantic Search |
| 2048       | Pending | |
| 4096       | Pending | May approach context window limits |
| 8192       | Pending | May exceed context window limits |

### Chunk Size Testing Results
*To be filled as tests complete* 