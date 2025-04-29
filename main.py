import json
import time
import asyncio
from typing import List, Dict
from ragas.integrations.r2r import transform_to_ragas_dataset
import logging
import colorlog
import os

from config import RAGConfig, RAGGenerationConfig, SearchSettings
from r2r_tester import RAGTester
from metrics import MetricsEvaluator, MetricsConfig
from chunking import LangChainChunking, AgenticChunking, ChunkingStrategy
from document import Document
from langchain_openai import ChatOpenAI
from generic_embeddings import GenericEmbeddings
from chunking import SemanticChunker, SDPMChunker

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

colors = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "red,bg_white",
}

formatter = colorlog.ColoredFormatter(
    "%(asctime)s - %(log_color)s%(levelname)s%(reset)s - %(message)s",
    log_colors=colors,
    reset=True,
    style="%",
)

console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


async def run_rag_evaluation(
    dataset_path: str,
    config: RAGConfig,
    metrics_config: MetricsConfig,
    chunker: ChunkingStrategy = LangChainChunking(),
    documents_dir: str = None,
    ingest_chunks: bool = True,
    delete_chunks: bool = True,
):
    """Main async function to run the complete RAG evaluation"""
    logger.info(f"Starting evaluation with config: {config}")

    # Initialize stats dictionary to track timing and metrics
    stats = {
        "start_time": time.time(),
        "chunking_strategy": chunker.__class__.__name__,
        "retrieval_approach": (
            config.search_settings.search_strategy
            if config.search_settings.search_strategy
            else "semantic_search"
        ),
        "document_count": 0,
        "total_chunks": 0,
        "avg_chunks_per_doc": 0,
        "delete_docs_time": 0,
        "chunking_time": 0,
        "ingestion_time": 0,
        "query_processing_time": 0,
        "evaluation_time": 0,
        "total_time": 0,
    }

    async with RAGTester() as tester:

        if delete_chunks:
            delete_start = time.time()
            await tester.delete_all_documents()
            stats["delete_docs_time"] = time.time() - delete_start

        logger.info(f"Loading dataset from {dataset_path}")
        with open(dataset_path, "r") as f:
            dataset = json.load(f)

        user_inputs = dataset["user_input"]
        references = dataset["reference"]
        reference_contexts = dataset["reference_contexts"]

        if ingest_chunks:
            chunking_start = time.time()
            total_chunks = 0

            if documents_dir:
                logger.info(f"Loading documents from {documents_dir}")
                files = [
                    f
                    for f in os.listdir(documents_dir)
                    if os.path.isfile(os.path.join(documents_dir, f))
                ]
                stats["document_count"] = len(files)
                logger.info(f"Found {len(files)} files in {documents_dir}")

                semaphore = asyncio.Semaphore(os.cpu_count() * 4)

                all_chunk_counts = []

                async def process_file(file):
                    nonlocal total_chunks
                    async with semaphore:
                        file_path = os.path.join(documents_dir, file)
                        logger.info(f"Processing document: {file}")

                        try:
                            with open(file_path, "r") as f:
                                document = Document(content=f.read(), name=file)

                            per_file_chunking_start = time.time()
                            chunks = await chunker.chunk_async(document)
                            chunk_count = len(chunks)
                            all_chunk_counts.append(chunk_count)

                            ingestion_start = time.time()
                            await tester.ingest_chunks(
                                [chunk.content for chunk in chunks]
                            )
                            ingestion_end = time.time()

                            # Track statistics
                            total_chunks += chunk_count

                            return {
                                "file": file,
                                "chunk_count": chunk_count,
                                "chunking_time": ingestion_start
                                - per_file_chunking_start,
                                "ingestion_time": ingestion_end - ingestion_start,
                            }

                        except Exception as e:
                            logger.error(f"Error processing {file}: {str(e)}")
                            return {"file": file, "error": str(e)}

                tasks = [process_file(file) for file in files]
                file_results = await asyncio.gather(*tasks)

                # Calculate statistics
                stats["chunking_time"] = time.time() - chunking_start
                stats["total_chunks"] = total_chunks
                if stats["document_count"] > 0:
                    stats["avg_chunks_per_doc"] = total_chunks / stats["document_count"]

                # Calculate ingestion time from file_results
                total_ingestion_time = sum(
                    result.get("ingestion_time", 0)
                    for result in file_results
                    if "ingestion_time" in result
                )
                stats["ingestion_time"] = total_ingestion_time

                # Store detailed per-file results
                stats["per_file_results"] = file_results

        query_start = time.time()
        r2r_responses = await tester.process_rag_queries(user_inputs, config)
        stats["query_processing_time"] = time.time() - query_start

        # Filter out user inputs, references, and contexts where we have no response
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

        logger.info(f"Evaluating {len(filtered_user_inputs)} results...")
        eval_start = time.time()
        ragas_eval_dataset = transform_to_ragas_dataset(
            user_inputs=filtered_user_inputs,
            r2r_responses=filtered_r2r_responses,
            references=filtered_references,
            reference_contexts=filtered_reference_contexts,
        )

        evaluator = MetricsEvaluator(metrics_config)
        results = evaluator.evaluate_dataset(ragas_eval_dataset)
        stats["evaluation_time"] = time.time() - eval_start

        logger.info(f"Evaluation results: {results}")

        # Calculate total time
        stats["total_time"] = time.time() - stats["start_time"]

        # Add the stats to the results
        results_with_stats = {
            "metrics": results,
            "stats": stats,
        }

        logger.info(f"Timing stats: {stats}")

        return results_with_stats


async def test_rag_configuration(
    dataset_path: str,
    documents_dir: str,
    config: RAGConfig,
    metrics_config: MetricsConfig,
    ingest_chunks: bool = True,
    delete_chunks: bool = True,
    chunker: ChunkingStrategy = LangChainChunking(),
):
    """Test a single RAG configuration asynchronously"""
    return await run_rag_evaluation(
        dataset_path=dataset_path,
        documents_dir=documents_dir,
        config=config,
        metrics_config=metrics_config,
        ingest_chunks=ingest_chunks,
        delete_chunks=delete_chunks,
        chunker=chunker,
    )


if __name__ == "__main__":

    generation_config = RAGGenerationConfig(
        model="deepseek/ISTA-DASLab/gemma-3-4b-it-GPTQ-4b-128g",
        api_base="http://172.18.21.136:8000/v1",
        temperature=0.1,
        max_tokens=50000,
    )

    """
    search_settings={
        "search_strategy": "hyde",
        "limit": 5,
        "use_hybrid_search": False,
        "graph_settings": {"enabled": False},
    }

    search_settings={
        "search_strategy": "rag_fusion",
        "limit": 5,
        "use_hybrid_search": False,
        "graph_settings": {"enabled": False},
    }
    """

    """
    search_settings = SearchSettings(
        limit=5,
        graph_settings={"enabled": False},
    )
    """

    search_setting_hyde = SearchSettings(
        search_strategy="hyde",
        use_hybrid_search=False,
        limit=5,
        graph_settings={"enabled": False},
    )

    search_setting_rag_fusion = SearchSettings(
        search_strategy="rag_fusion",
        use_hybrid_search=False,
        limit=5,
        graph_settings={"enabled": False},
    )

    search_settings_semantic_search = SearchSettings(
        limit=5,
        graph_settings={"enabled": False},
    )

    search_settings_hybrid_search = SearchSettings(
        limit=5,
        graph_settings={"enabled": False},
    )

    search_settings_with_names = [
        ("hyde", search_setting_hyde),
        ("rag_fusion", search_setting_rag_fusion),
        ("semantic_search", search_settings_semantic_search),
        ("hybrid_search", search_settings_hybrid_search),
    ]

    metrics_config = MetricsConfig(
        llm_model="ISTA-DASLab/gemma-3-4b-it-GPTQ-4b-128g",
        llm_api_base="http://172.18.21.136:8000/v1",
        embeddings_model="BAAI/bge-m3",
        embeddings_api_base="http://172.18.21.126:8000/v1",
    )

    # Define chunk sizes to test (doubling each time)
    chunk_sizes = [128, 256, 512, 1024, 2048, 4096, 8192]

    # Keep track of failures for tests.md
    chunk_size_results = {}

    # We'll use semantic search for chunk size testing as it had the best performance
    search_name = "semantic_search"
    search_setting = search_settings_semantic_search

    rag_config = RAGConfig(
        generation_config=generation_config,
        search_mode="basic",
        search_settings=search_setting,
        include_web_search=False,
    )

    # Test each chunk size
    for chunk_size in chunk_sizes:
        logger.info(f"======== Testing chunk size: {chunk_size} ========")
        word_size = chunk_size // 4  # Approximate words for semantic chunkers

        # Update chunk size result status
        chunk_size_results[chunk_size] = {"status": "Running", "notes": ""}

        chunkers = {
            "character": LangChainChunking(
                splitter_type="character",
                chunk_size=chunk_size,
                chunk_overlap=chunk_size // 4,
            ),
            "recursive": LangChainChunking(
                splitter_type="recursive",
                chunk_size=chunk_size,
                chunk_overlap=chunk_size // 4,
            ),
        }

        # Only include semantic and SDPM for sizes 32 and above
        if chunk_size >= 32:
            chunkers.update(
                {
                    "semantic": SemanticChunker(
                        embedding_model=GenericEmbeddings(
                            model="BAAI/bge-m3",
                            api_key="random_api_key",
                            base_url="http://172.18.21.126:8000/v1",
                            embedding_dimension=1024,
                        ),
                        chunk_size=word_size,  # This is in words
                        threshold=0.5,
                    ),
                    "Semantic_Double_Pass_Merging": SDPMChunker(
                        embedding_model=GenericEmbeddings(
                            model="BAAI/bge-m3",
                            api_key="random_api_key",
                            base_url="http://172.18.21.126:8000/v1",
                            embedding_dimension=1024,
                        ),
                        chunk_size=word_size,
                        threshold=0.5,
                    ),
                }
            )

        # Only include agentic for reasonable sizes (it's expensive for very small or large chunks)
        if chunk_size >= 32 and chunk_size <= 8192:
            chunkers["agentic"] = AgenticChunking(
                model=ChatOpenAI(
                    model="ISTA-DASLab/gemma-3-4b-it-GPTQ-4b-128gg",
                    base_url="http://172.18.21.136:8000/v1",
                    temperature=0.0,
                    max_tokens=50000,
                    api_key="random_api_key",
                ),
                max_chunk_size=chunk_size,
            )

        output_file = f"evaluation_results_{search_name}_chunk_size_{chunk_size}.json"
        results = {}
        stats_file = f"stats_{search_name}_chunk_size_{chunk_size}.json"
        stats = {}
        success = False
        failure_reason = ""

        try:
            for chunker_name, chunker in chunkers.items():
                logger.info(
                    f"Running evaluation with {chunker_name} chunker for chunk size {chunk_size}..."
                )

                max_retries = 3  # Reduced retries for chunk size testing
                retry_delay = 5

                chunker_success = False
                for attempt in range(max_retries):
                    try:
                        result = asyncio.run(
                            test_rag_configuration(
                                dataset_path="./datasets/ragas_generated_dataset.json",
                                documents_dir="./data/output",
                                config=rag_config,
                                metrics_config=metrics_config,
                                chunker=chunker,
                                ingest_chunks=True,
                                delete_chunks=True,
                            )
                        )
                        chunker_success = True
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            logger.warning(
                                f"Attempt {attempt + 1} failed for {chunker_name} with chunk size {chunk_size}: {str(e)}. Retrying in {retry_delay} seconds..."
                            )
                            time.sleep(retry_delay)
                        else:
                            logger.error(
                                f"All {max_retries} attempts failed for chunker {chunker_name} with chunk size {chunk_size}: {str(e)}"
                            )
                            # Just record the failure, don't raise the exception

                if chunker_success:
                    import ast

                    # Separate metrics and stats
                    result_dict = ast.literal_eval(str(result))

                    if (
                        isinstance(result_dict, dict)
                        and "metrics" in result_dict
                        and "stats" in result_dict
                    ):
                        results[chunker_name] = result_dict["metrics"]
                        stats[chunker_name] = result_dict["stats"]
                    else:
                        # For backward compatibility with previous result format
                        results[chunker_name] = result_dict
                        stats[chunker_name] = {"error": "No timing stats available"}
                else:
                    error_msg = f"Failed after {max_retries} attempts"
                    results[chunker_name] = {"error": error_msg}
                    stats[chunker_name] = {"error": error_msg}
                    logger.warning(
                        f"Skipping {chunker_name} for chunk size {chunk_size} due to errors"
                    )

            # Save results and stats to separate files
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)

            with open(stats_file, "w") as f:
                json.dump(stats, f, indent=2)

            logger.info(f"Results for chunk size {chunk_size} saved to {output_file}")
            logger.info(f"Statistics for chunk size {chunk_size} saved to {stats_file}")

            success = True

        except Exception as e:
            failure_reason = str(e)
            logger.error(
                f"Complete failure testing chunk size {chunk_size}: {failure_reason}"
            )

        # Update tests.md with status
        if success:
            chunk_size_results[chunk_size] = {
                "status": "Completed",
                "notes": f"Results saved to {output_file}, stats saved to {stats_file}",
            }
        else:
            chunk_size_results[chunk_size] = {
                "status": "Failed",
                "notes": f"Error: {failure_reason}",
            }

    # Update tests.md with results
    with open("tests.md", "r") as f:
        tests_md_content = f.read()

    # Update chunk size testing table
    chunk_size_table = (
        "| Chunk Size | Status | Notes |\n|------------|--------|-------|\n"
    )
    for size in sorted(chunk_size_results.keys()):
        result = chunk_size_results[size]
        chunk_size_table += f"| {size} | {result['status']} | {result['notes']} |\n"

    # Replace the existing table with the updated one
    import re

    pattern = r"## Chunk Size Testing.*?### Chunk Size Testing Results"
    replacement = f"## Chunk Size Testing\nTests with varying chunk sizes to determine optimal chunking parameters for each approach.\n\n{chunk_size_table}\n### Chunk Size Testing Results"
    tests_md_content = re.sub(pattern, replacement, tests_md_content, flags=re.DOTALL)

    with open("tests.md", "w") as f:
        f.write(tests_md_content)

    logger.info(
        "All chunk size tests completed. Results saved to individual files and tests.md updated."
    )

    # Previous testing code (commented out to focus on chunk size testing)
    """
    for search_name, search_setting in search_settings_with_names:
        logger.info(f"Running evaluation with {search_name} search setting...")

        if search_name == "semantic_search":
            rag_config = RAGConfig(
                generation_config=generation_config,
                search_mode="basic",
                search_settings=search_setting,
                include_web_search=False,
            )

        elif search_name == "hybrid_search":
            rag_config = RAGConfig(
                generation_config=generation_config,
                search_mode="advanced",
                search_settings=search_setting,
                include_web_search=False,
            )
        else:
            rag_config = RAGConfig(
                generation_config=generation_config,
                search_settings=search_setting,
                include_web_search=False,
            )

        output_file = f"evaluation_results_{search_name}.json"
        results = {}

        for chunker_name, chunker in chunkers.items():
            logger.info(
                f"Running evaluation with {chunker_name} chunker for {search_name}..."
            )

            max_retries = 10
            retry_delay = 5

            for attempt in range(max_retries):
                try:
                    result = asyncio.run(
                        test_rag_configuration(
                            dataset_path="./datasets/ragas_generated_dataset.json",
                            documents_dir="./data/output",
                            config=rag_config,
                            metrics_config=metrics_config,
                            chunker=chunker,
                            ingest_chunks=True,
                            delete_chunks=True,
                        )
                    )
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay} seconds..."
                        )
                        time.sleep(retry_delay)
                    else:
                        logger.error(
                            f"All {max_retries} attempts failed for chunker {chunker_name}"
                        )
                        raise

            import ast

            result = ast.literal_eval(str(result))
            results[chunker_name] = result

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results for {search_name} search setting saved to {output_file}")
    """
