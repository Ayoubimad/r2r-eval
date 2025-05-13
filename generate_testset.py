"""
Test dataset generation module for the RAG evaluation framework.

This module provides functionality to generate synthetic test datasets with questions, answers,
and reference contexts using RAGAS and various query synthesizers.
"""

import json
import os
from typing import List, Tuple, Optional, Any, Dict

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from ragas import RunConfig
from ragas.cache import DiskCacheBackend
from ragas.integrations.langchain import LangchainLLMWrapper, LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
)

from chunking import LangChainChunking
from utils import create_logger

logger = create_logger("testset_generator")

DEFAULT_CONFIG = {
    "DATA_DIR": "./data/output",
    "OUTPUT_PATH": "datasets/ragas_generated_dataset.json",
    "TESTSET_SIZE": 250,
    "TIMEOUT": 60000,
    "EMBEDDING_MODEL": "BAAI/bge-m3",
    "EMBEDDING_API_URL": "http://172.18.21.126:8000/v1",
    "LLM_MODEL": "ISTA-DASLab/gemma-3-4b-it-GPTQ-4b-128g",
    "LLM_API_URL": "http://172.18.21.138:8000/v1",
    "CACHE_DIR": "ragas_cache",
    "MAX_TOKENS": 10000,
    "WITH_DEBUGGING_LOGS": True,
    "TOP_P": 0.95,
    "TEMPERATURE": 0.8,
}


def load_documents(data_dir: str) -> List[Any]:
    """
    Load markdown documents from the data directory.

    Args:
        data_dir: Directory containing documents to load

    Returns:
        List of loaded documents
    """
    logger.info(f"Loading documents from {data_dir}")

    loader = DirectoryLoader(
        data_dir,
        glob="*.md",
        loader_cls=TextLoader,
    )

    documents = loader.load()
    logger.info(f"Loaded {len(documents)} documents")

    return documents


def process_documents(docs: List[Any]) -> List[Any]:
    """
    Clean the text in each document.

    Args:
        docs: List of documents to process

    Returns:
        Processed documents with clean text
    """
    logger.info("Processing documents")
    chunker = LangChainChunking()

    for doc in docs:
        doc.page_content = chunker.clean_text(doc.page_content)

    logger.info("Document processing complete")
    return docs


def setup_models(
    embedding_model: str,
    embedding_api_url: str,
    llm_model: str,
    llm_api_url: str,
    timeout: int,
    max_tokens: int,
    top_p: float,
    temperature: float,
    cache_dir: str,
) -> Tuple[LangchainLLMWrapper, LangchainEmbeddingsWrapper]:
    """
    Initialize and configure the embedding and LLM models.

    Args:
        embedding_model: Embedding model name
        embedding_api_url: Base URL for embedding API
        llm_model: LLM model name
        llm_api_url: Base URL for LLM API
        timeout: Request timeout in milliseconds
        max_tokens: Maximum tokens for LLM generation
        top_p: Top-p sampling parameter
        temperature: Temperature for LLM generation
        cache_dir: Directory for caching results

    Returns:
        Tuple of LLM wrapper and embeddings wrapper
    """
    logger.info(f"Setting up models: {embedding_model} and {llm_model}")

    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        api_key="random_api_key",
        base_url=embedding_api_url,
        timeout=timeout,
    )

    llm = ChatOpenAI(
        model=llm_model,
        api_key="random_api_key",
        base_url=llm_api_url,
        timeout=timeout,
        max_tokens=max_tokens,
        top_p=top_p,
        temperature=temperature,
    )

    cache = DiskCacheBackend(cache_dir=cache_dir)

    llm_wrapper = LangchainLLMWrapper(llm, cache=cache)
    embeddings_wrapper = LangchainEmbeddingsWrapper(embeddings, cache=cache)

    logger.info("Models setup complete")
    return llm_wrapper, embeddings_wrapper


def generate_testset(
    docs: List[Any],
    llm_wrapper: LangchainLLMWrapper,
    embeddings_wrapper: LangchainEmbeddingsWrapper,
    testset_size: int,
    timeout: int,
    with_debugging_logs: bool,
    query_distribution: Optional[List[Tuple[Any, float]]] = None,
) -> Any:
    """
    Generate a test dataset using RAGAS.

    Args:
        docs: List of documents to use for generation
        llm_wrapper: LLM wrapper for RAGAS
        embeddings_wrapper: Embeddings wrapper for RAGAS
        testset_size: Number of test cases to generate
        timeout: Request timeout in milliseconds
        with_debugging_logs: Whether to enable debugging logs
        query_distribution: Distribution of query types to generate

    Returns:
        Generated RAGAS dataset
    """
    logger.info(f"Generating test dataset with {len(docs)} documents")
    generator = TestsetGenerator(llm=llm_wrapper, embedding_model=embeddings_wrapper)

    run_config = RunConfig(timeout=timeout, max_workers=os.cpu_count() * 4)

    if query_distribution is None:
        # default query distribution
        query_distribution = [
            (SingleHopSpecificQuerySynthesizer(llm=llm_wrapper), 0.50),
            (MultiHopAbstractQuerySynthesizer(llm=llm_wrapper), 0.25),
            (MultiHopSpecificQuerySynthesizer(llm=llm_wrapper), 0.25),
        ]
        logger.info("Using default query distribution")
    else:
        logger.info("Using custom query distribution")

    logger.info(f"Starting generation of {testset_size} test cases")
    dataset = generator.generate_with_langchain_docs(
        documents=docs,
        query_distribution=query_distribution,
        run_config=run_config,
        testset_size=testset_size,
        with_debugging_logs=with_debugging_logs,
    )

    logger.info(f"Generated {len(dataset)} test cases")
    return dataset


def format_and_save_dataset(dataset: Any, output_path: str) -> None:
    """
    Convert dataset to the required format and save to disk.

    Args:
        dataset: RAGAS dataset to format and save
        output_path: Path where to save the formatted dataset
    """
    logger.info(f"Formatting dataset for saving to {output_path}")
    df = dataset.to_pandas()

    ragtester_dataset = {
        "user_input": df["user_input"].tolist(),
        "reference": df["reference"].tolist(),
        "reference_contexts": df["reference_contexts"].tolist(),
    }

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created directory {output_dir}")

    with open(output_path, "w") as f:
        json.dump(ragtester_dataset, f, indent=2)

    logger.info(f"Saved dataset with {len(df)} samples to {output_path}")


def generate_dataset(config: Optional[Dict[str, Any]] = None) -> str:
    """
    Main function to orchestrate the testset generation process.

    Args:
        config: Configuration dictionary with parameters

    Returns:
        Path to the generated dataset file
    """
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)

    logger.info(f"Starting dataset generation with configuration: {cfg}")

    docs = load_documents(cfg["DATA_DIR"])
    processed_docs = process_documents(docs)

    llm_wrapper, embeddings_wrapper = setup_models(
        embedding_model=cfg["EMBEDDING_MODEL"],
        embedding_api_url=cfg["EMBEDDING_API_URL"],
        llm_model=cfg["LLM_MODEL"],
        llm_api_url=cfg["LLM_API_URL"],
        timeout=cfg["TIMEOUT"],
        max_tokens=cfg["MAX_TOKENS"],
        top_p=cfg["TOP_P"],
        temperature=cfg["TEMPERATURE"],
        cache_dir=cfg["CACHE_DIR"],
    )

    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=llm_wrapper), 0.50),
        (MultiHopAbstractQuerySynthesizer(llm=llm_wrapper), 0.25),
        (MultiHopSpecificQuerySynthesizer(llm=llm_wrapper), 0.25),
    ]

    dataset = generate_testset(
        docs=processed_docs,
        llm_wrapper=llm_wrapper,
        embeddings_wrapper=embeddings_wrapper,
        testset_size=cfg["TESTSET_SIZE"],
        timeout=cfg["TIMEOUT"],
        with_debugging_logs=cfg["WITH_DEBUGGING_LOGS"],
        query_distribution=query_distribution,
    )

    format_and_save_dataset(dataset, cfg["OUTPUT_PATH"])

    logger.info("Dataset generation complete")
    return cfg["OUTPUT_PATH"]


if __name__ == "__main__":
    output_path = generate_dataset()
    print(f"Dataset successfully generated at: {output_path}")
