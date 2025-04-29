# RAG Evaluation Project Recap

## Project Overview
We are evaluating a Retrieval Augmented Generation (RAG) system with various configurations to determine optimal settings for different use cases. The evaluation systematically tests different retrieval approaches and chunking strategies while measuring performance on standardized metrics.

## What We've Done So Far

### 1. Initial Testing
- Evaluated four retrieval approaches: Semantic Search, RAG Fusion, Hybrid Search, and HyDE
- Tested five chunking strategies: Character, Agentic, Semantic, SDPM, and Recursive
- Used a fixed chunk size of 1024 characters for the initial comparison
- Discovered that Semantic Search performs best overall, especially with Semantic/SDPM chunking strategies

### 2. Visualization Development
- Created comprehensive visualization scripts to analyze results
- Added multiple chart types (radar charts, bar charts, heatmaps, precision-recall curves, etc.)
- Enhanced visualizations with annotations, insights, and performance metrics
- Implemented specific visualizations to focus on precision-recall trade-offs

### 3. Documentation 
- Created tests.md to document test configurations and results
- Recorded key observations and insights from our analyses
- Set up structure for documenting future test results

### 4. Chunk Size Testing
- Modified the main.py script to systematically test different chunk sizes (128-8192)
- Implemented error handling to allow testing to continue even if some configurations fail
- Set up automatic documentation of results in tests.md

## Key Findings
1. Semantic Search consistently outperforms other retrieval approaches in terms of faithfulness and precision
2. SDPM and Semantic chunking strategies generally yield the best results
3. There's a clear trade-off between precision and recall, with Semantic Search offering the best balance
4. Answer relevancy is consistently low across approaches, suggesting an area for further optimization

## Current Testing Status
- Initial evaluation of retrieval approaches and chunking strategies is complete
- Visualization framework is in place
- Chunk size testing is ready to run
- Testing different chunk sizes will help determine the optimal size for each approach/strategy

## Next Steps
1. Run the chunk size tests to find the optimal chunk size for each approach
2. Analyze results to understand the relationship between chunk size and performance
3. Consider testing with different context window limits
4. Potentially explore ways to improve answer relevancy, which is currently low across all approaches

## System Configuration
- Using models:
  - LLM: deepseek/ISTA-DASLab/gemma-3-4b-it-GPTQ-4b-128g
  - Embeddings: BAAI/bge-m3
- Testing with a limit of 5 chunks per query
- Various chunking strategies with configurable parameters 