"""
Main entry point for the RAG evaluation framework.

This module provides the main functionality for running evaluations of
different RAG configurations, chunking strategies, and search approaches.
"""

import json
import time
import asyncio
import os
from typing import List, Dict, Any, Optional, Tuple, Set
import ast

from ragas.integrations.r2r import transform_to_ragas_dataset

from config import RAGConfig, RAGGenerationConfig, SearchSettings, MetricsConfig
from r2r_tester import RAGTester
from metrics import MetricsEvaluator
from chunking import (
    LangChainChunking,
    AgenticChunking,
    ChunkingStrategy,
    SemanticChunker,
    SDPMChunker,
)
from chunk_enrichment import ChunkEnrichmentStrategy
from document import Document
from langchain_openai import ChatOpenAI
from generic_embeddings import GenericEmbeddings
from utils import create_logger

logger = create_logger("main")


async def run_rag_evaluation(
    dataset_path: str,
    config: RAGConfig,
    metrics_config: MetricsConfig,
    chunker: ChunkingStrategy,
    documents_dir: Optional[str] = None,
    ingest_chunks: bool = True,
    delete_chunks: bool = True,
    enrich_chunks: bool = False,
    enrichment_strategy: Optional[ChunkEnrichmentStrategy] = None,
) -> Dict[str, Any]:
    """
    Main async function to run the complete RAG evaluation.

    Args:
        dataset_path: Path to the dataset file
        config: RAG configuration
        metrics_config: Metrics evaluation configuration
        chunker: Chunking strategy to use
        documents_dir: Directory containing documents to process
        ingest_chunks: Whether to ingest chunks into the RAG system
        delete_chunks: Whether to delete existing chunks before ingestion
        enrich_chunks: Whether to enrich chunks with context
        enrichment_strategy: Strategy for chunk enrichment

    Returns:
        Dictionary containing evaluation results
    """
    async with RAGTester() as tester:
        if delete_chunks:
            await tester.delete_all_documents()

        dataset = load_dataset(dataset_path)
        user_inputs, references, reference_contexts = extract_dataset_components(
            dataset
        )

        if ingest_chunks and documents_dir:
            await process_and_ingest_documents(
                tester,
                documents_dir,
                chunker,
                enrich_chunks=enrich_chunks,
                enrichment_strategy=enrichment_strategy,
            )

        r2r_responses = await tester.process_rag_queries(user_inputs, config)

        filtered_inputs, filtered_refs, filtered_contexts, filtered_responses = (
            filter_failed_queries(
                user_inputs, references, reference_contexts, r2r_responses
            )
        )

        logger.info(f"Evaluating {len(filtered_inputs)} results...")

        results = evaluate_responses(
            filtered_inputs,
            filtered_responses,
            filtered_refs,
            filtered_contexts,
            metrics_config,
        )

        logger.info(f"Evaluation results: {results}")

        return results


async def process_and_ingest_documents(
    tester: RAGTester,
    documents_dir: str,
    chunker: ChunkingStrategy,
    enrich_chunks: bool = False,
    enrichment_strategy: Optional[ChunkEnrichmentStrategy] = None,
) -> None:
    """
    Process documents and ingest chunks into the RAG system.

    Args:
        tester: RAG tester instance
        documents_dir: Directory containing documents to process
        chunker: Chunking strategy to use
        enrich_chunks: Whether to enrich chunks with context
        enrichment_strategy: Strategy for chunk enrichment
    """
    logger.info(f"Loading documents from {documents_dir}")
    files = [
        f
        for f in os.listdir(documents_dir)
        if os.path.isfile(os.path.join(documents_dir, f))
    ]
    logger.info(f"Found {len(files)} files in {documents_dir}")

    semaphore = asyncio.Semaphore(16)

    async def process_file(file):
        async with semaphore:
            file_path = os.path.join(documents_dir, file)
            logger.info(f"Processing document: {file}")

            try:
                with open(file_path, "r") as f:
                    document = Document(content=f.read(), name=file)

                chunks = await chunker.chunk_async(document)
                chunk_contents = [chunk.content for chunk in chunks]

                if enrich_chunks and enrichment_strategy:
                    logger.info(
                        f"Enriching {len(chunk_contents)} chunks for document: {file}"
                    )
                    chunk_contents = await enrichment_strategy.enrich_chunks_async(
                        chunk_contents
                    )

                await tester.ingest_chunks(chunk_contents)

            except Exception as e:
                logger.error(f"Error processing {file}: {str(e)}")

    tasks = [process_file(file) for file in files]
    await asyncio.gather(*tasks)


def load_dataset(dataset_path: str) -> Dict[str, Any]:
    """
    Load the evaluation dataset from a file.

    Args:
        dataset_path: Path to the dataset file

    Returns:
        Loaded dataset
    """
    logger.info(f"Loading dataset from {dataset_path}")
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    return dataset


def extract_dataset_components(
    dataset: Dict[str, Any],
) -> Tuple[List[str], List[str], List[str]]:
    """
    Extract components from the dataset.

    Args:
        dataset: Dataset dictionary

    Returns:
        Tuple of (user_inputs, references, reference_contexts)
    """
    return (
        dataset["user_input"],
        dataset["reference"],
        dataset["reference_contexts"],
    )


def filter_failed_queries(
    user_inputs: List[str],
    references: List[str],
    reference_contexts: List[List[str]],
    r2r_responses: List[Optional[str]],
) -> Tuple[List[str], List[str], List[List[str]], List[str]]:
    """
    Filter out failed queries from the evaluation data.

    Args:
        user_inputs: List of user input queries
        references: List of reference answers
        reference_contexts: List of reference context sets
        r2r_responses: List of RAG system responses

    Returns:
        Tuple of filtered (user_inputs, references, reference_contexts, responses)
    """
    valid_indices = [i for i, resp in enumerate(r2r_responses) if resp]

    if len(valid_indices) < len(user_inputs):
        logger.warning(
            f"Filtering out {len(user_inputs) - len(valid_indices)} failed queries from evaluation"
        )
        filtered_user_inputs = [user_inputs[i] for i in valid_indices]
        filtered_references = [references[i] for i in valid_indices]
        filtered_reference_contexts = [reference_contexts[i] for i in valid_indices]
        filtered_r2r_responses = [r for r in r2r_responses if r]
    else:
        filtered_user_inputs = user_inputs
        filtered_references = references
        filtered_reference_contexts = reference_contexts
        filtered_r2r_responses = r2r_responses

    return (
        filtered_user_inputs,
        filtered_references,
        filtered_reference_contexts,
        filtered_r2r_responses,
    )


def evaluate_responses(
    user_inputs: List[str],
    r2r_responses: List[str],
    references: List[str],
    reference_contexts: List[List[str]],
    metrics_config: MetricsConfig,
) -> Dict[str, Any]:
    """
    Evaluate RAG responses using RAGAS metrics.

    Args:
        user_inputs: List of user input queries
        r2r_responses: List of RAG system responses
        references: List of reference answers
        reference_contexts: List of reference context sets
        metrics_config: Metrics evaluation configuration

    Returns:
        Dictionary of evaluation results
    """
    ragas_eval_dataset = transform_to_ragas_dataset(
        user_inputs=user_inputs,
        r2r_responses=r2r_responses,
        references=references,
        reference_contexts=reference_contexts,
    )

    evaluator = MetricsEvaluator(metrics_config)
    return evaluator.evaluate_dataset(ragas_eval_dataset)


async def test_rag_configuration(
    dataset_path: str,
    documents_dir: str,
    config: RAGConfig,
    metrics_config: MetricsConfig,
    ingest_chunks: bool = True,
    delete_chunks: bool = True,
    chunker: ChunkingStrategy = LangChainChunking(),
    enrich_chunks: bool = False,
    enrichment_strategy: Optional[ChunkEnrichmentStrategy] = None,
) -> Dict[str, Any]:
    """
    Test a single RAG configuration asynchronously.

    Args:
        dataset_path: Path to the dataset file
        documents_dir: Directory containing documents to process
        config: RAG configuration
        metrics_config: Metrics evaluation configuration
        ingest_chunks: Whether to ingest chunks into the RAG system
        delete_chunks: Whether to delete existing chunks before ingestion
        chunker: Chunking strategy to use
        enrich_chunks: Whether to enrich chunks with context
        enrichment_strategy: Strategy for chunk enrichment

    Returns:
        Dictionary containing evaluation results
    """
    return await run_rag_evaluation(
        dataset_path=dataset_path,
        documents_dir=documents_dir,
        config=config,
        metrics_config=metrics_config,
        ingest_chunks=ingest_chunks,
        delete_chunks=delete_chunks,
        chunker=chunker,
        enrich_chunks=enrich_chunks,
        enrichment_strategy=enrichment_strategy,
    )


def create_chunker(
    chunker_type: str,
    chunk_size: int,
    embedding_base_url: str,
    model_base_url: str,
) -> ChunkingStrategy:
    """
    Create a chunker instance based on the chunker type and settings.

    Args:
        chunker_type: Type of chunker to create
        chunk_size: Size of chunks to create
        embedding_base_url: Base URL for embedding API
        model_base_url: Base URL for LLM API

    Returns:
        Configured chunker instance
    """
    word_size = chunk_size // 4  # Approximate words for semantic chunkers
    chunk_overlap = chunk_size // 4  # Standard 25% overlap

    if chunker_type == "character":
        chunker = LangChainChunking(
            splitter_type="character",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        setattr(chunker, "chunker_type", "character")
        return chunker
    elif chunker_type == "recursive":
        chunker = LangChainChunking(
            splitter_type="recursive",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        setattr(chunker, "chunker_type", "recursive")
        return chunker
    elif chunker_type == "semantic":
        chunker = SemanticChunker(
            embedding_model=GenericEmbeddings(
                model="BAAI/bge-m3",
                api_key="random_api_key",
                base_url=embedding_base_url,
                embedding_dimension=1024,
            ),
            chunk_size=word_size,  # This is in words
            threshold=0.5,
        )
        setattr(chunker, "chunker_type", "semantic")
        return chunker
    elif chunker_type == "sdpm":
        chunker = SDPMChunker(
            embedding_model=GenericEmbeddings(
                model="BAAI/bge-m3",
                api_key="random_api_key",
                base_url=embedding_base_url,
                embedding_dimension=1024,
            ),
            chunk_size=word_size,
            threshold=0.5,
        )
        setattr(chunker, "chunker_type", "sdpm")
        return chunker
    elif chunker_type == "agentic":
        chunker = AgenticChunking(
            model=ChatOpenAI(
                model="ISTA-DASLab/gemma-3-4b-it-GPTQ-4b-128g",
                base_url=model_base_url,
                temperature=0.0,
                max_tokens=50000,
                api_key="random_api_key",
            ),
            max_chunk_size=chunk_size,
        )
        setattr(chunker, "chunker_type", "agentic")
        return chunker
    else:
        raise ValueError(f"Unknown chunker type: {chunker_type}")


def run_chunk_size_test(
    chunk_size: int,
    search_settings_dict: Dict[str, Dict[str, Any]],
    chunker_types: Set[str],
    generation_config: RAGGenerationConfig,
    metrics_config: MetricsConfig,
    output_prefix: str,
    documents_dir: str,
    dataset_path: str,
    max_retries: int = 3,
    enrich_chunks: bool = False,
    enrichment_strategy: ChunkEnrichmentStrategy = None,
) -> Dict[str, Any]:
    """
    Run tests for a specific chunk size across multiple chunker types and all search strategies.
    """
    chunkers = {}

    if "character" in chunker_types:
        chunker = create_chunker(
            "character",
            chunk_size,
            metrics_config.embeddings_api_base,
            generation_config.api_base,
        )
        setattr(chunker, "chunker_type", "character")
        chunkers["character"] = chunker

    if "recursive" in chunker_types:
        chunker = create_chunker(
            "recursive",
            chunk_size,
            metrics_config.embeddings_api_base,
            generation_config.api_base,
        )
        setattr(chunker, "chunker_type", "recursive")
        chunkers["recursive"] = chunker

    if "semantic" in chunker_types:
        chunker = create_chunker(
            "semantic",
            chunk_size,
            metrics_config.embeddings_api_base,
            generation_config.api_base,
        )
        setattr(chunker, "chunker_type", "semantic")
        chunkers["semantic"] = chunker

    if "sdpm" in chunker_types:
        chunker = create_chunker(
            "sdpm",
            chunk_size,
            metrics_config.embeddings_api_base,
            generation_config.api_base,
        )
        setattr(chunker, "chunker_type", "sdpm")
        chunkers["Semantic_Double_Pass_Merging"] = chunker

    if "agentic" in chunker_types:
        chunker = create_chunker(
            "agentic",
            chunk_size,
            metrics_config.embeddings_api_base,
            generation_config.api_base,
        )
        setattr(chunker, "chunker_type", "agentic")
        chunkers["agentic"] = chunker

    output_file = f"{output_prefix}_{chunk_size}.json"
    all_results = {}

    try:
        for chunker_name, chunker in chunkers.items():
            logger.info(f'Ingesting documents with chunker "{chunker_name}"')
            # Ingest chunks ONCE for this chunker, after deleting previous chunks
            ingestion_success = False
            for attempt in range(max_retries):
                try:
                    # Use a dummy config, only for ingestion
                    dummy_config = RAGConfig(
                        generation_config=generation_config,
                        search_settings=list(search_settings_dict.values())[0][
                            "config"
                        ],
                        search_mode=list(search_settings_dict.values())[0]["mode"],
                    )

                    async def delete_and_ingest():
                        async with RAGTester() as tester:
                            await tester.delete_all_documents()
                            await process_and_ingest_documents(
                                tester=tester,
                                documents_dir=documents_dir,
                                chunker=chunker,
                                enrich_chunks=enrich_chunks,
                                enrichment_strategy=enrichment_strategy,
                            )

                    import nest_asyncio

                    nest_asyncio.apply()
                    asyncio.run(delete_and_ingest())
                    ingestion_success = True
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(5)
                    else:
                        logger.error(
                            f"All {max_retries} attempts failed for ingestion with {chunker_name}: {str(e)}"
                        )
                        all_results[chunker_name] = {
                            "error": f"Ingestion failed: {str(e)}"
                        }
                        break
            if not ingestion_success:
                continue  # Skip to next chunker if ingestion failed

            logger.info(
                f'Starting RAG with search strategies: {", ".join(search_settings_dict.keys())}'
            )
            # Now run all search strategies WITHOUT re-ingesting
            chunker_results = {}
            for search_name, search_config in search_settings_dict.items():
                logger.info(f'Starting RAG with "{search_name}"')
                rag_config = RAGConfig(
                    generation_config=generation_config,
                    search_settings=search_config["config"],
                    search_mode=search_config["mode"],
                )
                for attempt in range(max_retries):
                    try:
                        result = asyncio.run(
                            test_rag_configuration(
                                dataset_path=dataset_path,
                                documents_dir=documents_dir,
                                config=rag_config,
                                metrics_config=metrics_config,
                                chunker=chunker,
                                ingest_chunks=False,
                                delete_chunks=False,
                                enrich_chunks=enrich_chunks,
                                enrichment_strategy=enrichment_strategy,
                            )
                        )
                        result_dict = ast.literal_eval(str(result))
                        chunker_results[search_name] = result_dict
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            time.sleep(5)
                        else:
                            logger.error(
                                f"All {max_retries} attempts failed for {chunker_name} with search {search_name}: {str(e)}"
                            )
                            chunker_results[search_name] = {
                                "error": f"Failed after {max_retries} attempts: {str(e)}"
                            }
            all_results[chunker_name] = chunker_results

        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        return all_results

    except Exception as e:
        logger.error(f"Complete failure testing chunk size {chunk_size}: {str(e)}")
        return {"error": str(e)}


def main():
    """Main entry point for the RAG evaluation framework."""

    documents_dir = "/home/e4user/r2r-eval/data/pdfservers_md"
    dataset_path = "/home/e4user/r2r-eval/datasets/ragas_testset_dell_servers.json"
    output_prefix = "evaluation_results_dell_servers"

    generation_config = RAGGenerationConfig(
        model="deepseek/ISTA-DASLab/gemma-3-4b-it-GPTQ-4b-128g",
        api_base="http://172.18.21.136:8000/v1",
        temperature=0.1,
        max_tokens=50000,
    )

    search_settings = {
        "semantic_search": {
            "config": SearchSettings(
                limit=5,
                graph_settings={"enabled": False},
            ),
            "mode": "basic",
        },
        "hybrid_search": {
            "config": SearchSettings(
                use_hybrid_search=True,
                limit=5,
                graph_settings={"enabled": False},
            ),
            "mode": "advanced",
        },
        "rag_fusion": {
            "config": SearchSettings(
                search_strategy="rag_fusion",
                use_hybrid_search=False,
                limit=5,
                graph_settings={"enabled": False},
            ),
            "mode": None,
        },
        "hyde": {
            "config": SearchSettings(
                search_strategy="hyde",
                use_hybrid_search=False,
                limit=5,
                graph_settings={"enabled": False},
            ),
            "mode": None,
        },
    }

    metrics_config = MetricsConfig(
        llm_model="ISTA-DASLab/gemma-3-4b-it-GPTQ-4b-128g",
        llm_api_base="http://172.18.21.136:8000/v1",
        embeddings_model="BAAI/bge-m3",
        embeddings_api_base="http://172.18.21.126:8000/v1",
    )

    chunker_types = {
        "semantic",
        "sdpm",
        "agentic",
        "character",
        "recursive",
    }

    enrichment_strategy = ChunkEnrichmentStrategy(
        model="ISTA-DASLab/gemma-3-4b-it-GPTQ-4b-128g",
        api_base="http://172.18.21.136:8000/v1",
        temperature=0.8,
        max_tokens=8192,
        concurrency_limit=32,
        n_chunks=2,
    )

    enable_chunk_enrichment = False

    chunk_sizes = [1024]

    for chunk_size in chunk_sizes:
        run_chunk_size_test(
            chunk_size=chunk_size,
            search_settings_dict=search_settings,
            chunker_types=chunker_types,
            generation_config=generation_config,
            metrics_config=metrics_config,
            output_prefix=output_prefix,
            documents_dir=documents_dir,
            dataset_path=dataset_path,
            max_retries=3,
            enrich_chunks=enable_chunk_enrichment,
            enrichment_strategy=enrichment_strategy,
        )


if __name__ == "__main__":
    main()
