import sys
import json
import os

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

DATA_DIR = "./data/output"
OUTPUT_PATH = "datasets/ragas_generated_dataset_2.json"
TESTSET_SIZE = 250
TIMEOUT = 60000
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_API_URL = "http://172.18.21.126:8000/v1"
LLM_MODEL = "ISTA-DASLab/gemma-3-4b-it-GPTQ-4b-128g"
LLM_API_URL = "http://172.18.21.138:8000/v1"
CACHE_DIR = "ragas_cache"
MAX_TOKENS = 10000
WITH_DEBUGGING_LOGS = True
TOP_P = 0.95
TEMPERATURE = 0.8


def load_documents():
    """Load markdown documents from the data directory."""
    loader = DirectoryLoader(
        DATA_DIR,
        glob="*.md",
        loader_cls=TextLoader,
    )
    return loader.load()


def process_documents(docs):
    """Clean the text in each document."""
    chunker = LangChainChunking()

    for doc in docs:
        doc.page_content = chunker.clean_text(doc.page_content)

    return docs


def setup_models():
    """Initialize and configure the embedding and LLM models."""
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key="random_api_key",
        base_url=EMBEDDING_API_URL,
        timeout=TIMEOUT,
    )

    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key="random_api_key",
        base_url=LLM_API_URL,
        timeout=TIMEOUT,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
        temperature=TEMPERATURE,
    )

    cache = DiskCacheBackend(cache_dir=CACHE_DIR)

    llm_wrapper = LangchainLLMWrapper(llm, cache=cache)
    embeddings_wrapper = LangchainEmbeddingsWrapper(embeddings, cache=cache)

    return llm_wrapper, embeddings_wrapper


def generate_testset(docs, llm_wrapper, embeddings_wrapper, query_distribution):
    """Generate a test dataset using RAGAS."""
    generator = TestsetGenerator(llm=llm_wrapper, embedding_model=embeddings_wrapper)

    run_config = RunConfig(timeout=TIMEOUT, max_workers=os.cpu_count() * 4)

    if query_distribution is None:
        # default query distribution
        query_distribution = [
            (SingleHopSpecificQuerySynthesizer(llm=llm_wrapper), 0.50),
            (MultiHopAbstractQuerySynthesizer(llm=llm_wrapper), 0.25),
            (MultiHopSpecificQuerySynthesizer(llm=llm_wrapper), 0.25),
        ]

    dataset = generator.generate_with_langchain_docs(
        documents=docs,
        query_distribution=query_distribution,
        run_config=run_config,
        testset_size=TESTSET_SIZE,
        with_debugging_logs=WITH_DEBUGGING_LOGS,
    )

    return dataset


def format_and_save_dataset(dataset):
    """Convert dataset to the required format and save to disk."""
    df = dataset.to_pandas()

    ragtester_dataset = {
        "user_input": df["user_input"].tolist(),
        "reference": df["reference"].tolist(),
        "reference_contexts": df["reference_contexts"].tolist(),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(ragtester_dataset, f, indent=2)

    print(f"Saved dataset to {OUTPUT_PATH}")


def main():
    """Main function to orchestrate the testset generation process."""
    docs = load_documents()

    processed_docs = process_documents(docs)

    llm_wrapper, embeddings_wrapper = setup_models()

    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=llm_wrapper), 0.50),
        (MultiHopAbstractQuerySynthesizer(llm=llm_wrapper), 0.25),
        (MultiHopSpecificQuerySynthesizer(llm=llm_wrapper), 0.25),
    ]

    dataset = generate_testset(
        processed_docs, llm_wrapper, embeddings_wrapper, query_distribution
    )

    format_and_save_dataset(dataset)


if __name__ == "__main__":
    main()
