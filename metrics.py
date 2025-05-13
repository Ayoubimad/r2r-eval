"""
Metrics evaluation module for the RAG evaluation framework.

This module provides classes and utilities for evaluating RAG system performance
using various metrics from the Ragas framework.
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
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

from config import MetricsConfig
from utils import create_logger

logger = create_logger("metrics_evaluator")


class LLMFactory:
    """Factory for creating language models for evaluation"""

    @staticmethod
    def create_llm(
        model: str,
        api_key: str,
        api_base: str,
        temperature: float = 0.1,
        cache: Optional[DiskCacheBackend] = None,
    ) -> LangchainLLMWrapper:
        """
        Create a language model wrapped for Ragas

        Args:
            model: Model identifier
            api_key: API key for the model
            api_base: Base URL for the API
            temperature: Model temperature setting
            cache: Optional cache backend

        Returns:
            Wrapped language model
        """
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=api_base,
            temperature=temperature,
            top_p=1,
            timeout=60000,
            max_tokens=55000,
        )
        return LangchainLLMWrapper(llm, cache=cache)


class EmbeddingFactory:
    """Factory for creating embeddings models for evaluation"""

    @staticmethod
    def create_embeddings(
        model: str,
        api_key: str,
        api_base: str,
        cache: Optional[DiskCacheBackend] = None,
    ) -> LangchainEmbeddingsWrapper:
        """
        Create an embeddings model wrapped for Ragas

        Args:
            model: Model identifier
            api_key: API key for the model
            api_base: Base URL for the API
            cache: Optional cache backend

        Returns:
            Wrapped embeddings model
        """
        embeddings = OpenAIEmbeddings(
            model=model,
            api_key=api_key,
            base_url=api_base,
            timeout=60000,
        )
        return LangchainEmbeddingsWrapper(embeddings, cache=cache)


class MetricsFactory:
    """Factory for creating evaluation metrics"""

    @staticmethod
    def create_metrics(metric_names: Optional[List[str]] = None) -> List[Any]:
        """
        Create a list of metrics based on provided names

        Args:
            metric_names: Names of metrics to create (if None, creates all)

        Returns:
            List of metric instances
        """
        available_metrics = {
            "faithfulness": Faithfulness,
            "response_relevancy": ResponseRelevancy,
            "context_precision": ContextPrecision,
            "context_recall": ContextRecall,
        }

        if not metric_names:
            return [metric_class() for metric_class in available_metrics.values()]

        metrics = []
        for name in metric_names:
            name = name.lower()
            if name in available_metrics:
                metrics.append(available_metrics[name]())
            else:
                logger.warning(f"Unknown metric: {name}")

        return metrics or [
            metric_class() for metric_class in available_metrics.values()
        ]


class CacheManager:
    """Manager for Ragas cache"""

    def __init__(self, cache_dir: str = "ragas_cache"):
        """
        Initialize cache manager

        Args:
            cache_dir: Directory for storing cache
        """
        self.cache_dir = cache_dir
        self._cache = None

    @property
    def cache(self) -> DiskCacheBackend:
        """
        Get or create cache instance

        Returns:
            Cache backend
        """
        if self._cache is None:
            self._cache = DiskCacheBackend(cache_dir=self.cache_dir)
        return self._cache


class MetricsEvaluator:
    """Handles evaluation of RAG responses using Ragas metrics"""

    def __init__(self, config: MetricsConfig, cache_dir: str = "ragas_cache"):
        """
        Initialize the metrics evaluator

        Args:
            config: Configuration for metrics evaluation
            cache_dir: Directory for storing cache
        """
        self.config = config
        self.cache_manager = CacheManager(cache_dir)
        self.llm = self._setup_llm()
        self.embeddings = self._setup_embeddings()
        self.metrics = self._setup_metrics()

    def _setup_llm(self) -> LangchainLLMWrapper:
        """Initialize the language model for evaluation"""
        return LLMFactory.create_llm(
            model=self.config.llm_model,
            api_key=self.config.api_key_llm,
            api_base=self.config.llm_api_base,
            cache=self.cache_manager.cache,
        )

    def _setup_embeddings(self) -> LangchainEmbeddingsWrapper:
        """Initialize the embeddings model"""
        return EmbeddingFactory.create_embeddings(
            model=self.config.embeddings_model,
            api_key=self.config.api_key_embeddings,
            api_base=self.config.embeddings_api_base,
            cache=self.cache_manager.cache,
        )

    def _setup_metrics(self) -> List[Any]:
        """Initialize the evaluation metrics"""
        return MetricsFactory.create_metrics()

    def _create_run_config(self) -> RunConfig:
        """Create Ragas run configuration"""
        return RunConfig(
            timeout=60000,
            max_workers=os.cpu_count() * 3,
        )

    def evaluate_dataset(self, dataset: Any) -> Dict[str, float]:
        """
        Evaluate a dataset using Ragas metrics

        Args:
            dataset: A Ragas-compatible dataset

        Returns:
            Dict containing scores for each metric
        """
        try:
            run_config = self._create_run_config()
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
