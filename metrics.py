import os
from dataclasses import dataclass
from typing import List, Dict, Any
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas import evaluate
from ragas.cache import DiskCacheBackend
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas import RunConfig
from config import setup_logger

logger = setup_logger()


@dataclass
class MetricsConfig:
    """Configuration for metrics evaluation"""

    llm_model: str
    llm_api_base: str
    embeddings_model: str
    embeddings_api_base: str
    api_key: str = "random_api_key"


class MetricsEvaluator:
    """Handles evaluation of RAG responses using Ragas metrics"""

    def __init__(self, config: MetricsConfig):
        self.config = config
        self.cache = DiskCacheBackend(cache_dir="ragas_cache")
        self.llm = self._setup_llm()
        self.embeddings = self._setup_embeddings()
        self.metrics = self._setup_metrics()

    def _setup_llm(self) -> LangchainLLMWrapper:
        """Initialize the language model for evaluation"""
        llm = ChatOpenAI(
            model=self.config.llm_model,
            api_key=self.config.api_key,
            base_url=self.config.llm_api_base,
            temperature=0.1,
            top_p=1,
            timeout=60000,
            max_tokens=55000,
        )
        return LangchainLLMWrapper(llm, cache=self.cache)

    def _setup_embeddings(self) -> LangchainEmbeddingsWrapper:
        """Initialize the embeddings model"""
        embeddings = OpenAIEmbeddings(
            model=self.config.embeddings_model,
            api_key=self.config.api_key,
            base_url=self.config.embeddings_api_base,
            timeout=60000,
        )
        return LangchainEmbeddingsWrapper(embeddings, cache=self.cache)

    def _setup_metrics(self) -> List[Any]:
        """Initialize the evaluation metrics"""
        return [
            Faithfulness(),
            ResponseRelevancy(),
            ContextPrecision(),
            ContextRecall(),
        ]

    def evaluate_dataset(self, dataset: Any) -> Dict[str, float]:
        """
        Evaluate a dataset using Ragas metrics

        Args:
            dataset: A Ragas-compatible dataset

        Returns:
            Dict containing scores for each metric
        """
        try:
            run_config = RunConfig(timeout=60000, max_workers=os.cpu_count() * 4)
            results = evaluate(
                llm=self.llm,
                embeddings=self.embeddings,
                dataset=dataset,
                metrics=self.metrics,
                run_config=run_config,
                batch_size=500,
            )
            return results
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise
