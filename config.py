"""
Configuration module for the RAG evaluation framework.

This module provides data classes for configuration settings used throughout
the evaluation framework, ensuring type safety and validation.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Literal, List


@dataclass
class RAGGenerationConfig:
    """Configuration for the language model generation part of RAG"""

    model: str
    api_base: str
    temperature: float = 0.7
    max_tokens: int = 8192
    stream: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format for API calls"""
        return {
            "model": self.model,
            "api_base": self.api_base,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": self.stream,
        }


@dataclass
class SearchSettings:
    """Configuration for the search part of RAG"""

    limit: int = 10
    search_strategy: Optional[str] = None
    use_hybrid_search: bool = False
    use_semantic_search: bool = True
    filters: Optional[Dict[str, Any]] = None
    graph_settings: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format for API calls"""
        settings = {
            "limit": self.limit,
            "use_hybrid_search": self.use_hybrid_search,
            "use_semantic_search": self.use_semantic_search,
        }

        if self.search_strategy:
            settings["search_strategy"] = self.search_strategy
        if self.filters:
            settings["filters"] = self.filters
        if self.graph_settings:
            settings["graph_settings"] = self.graph_settings

        return settings


@dataclass
class RAGConfig:
    """Configuration for RAG system testing"""

    generation_config: RAGGenerationConfig
    search_settings: SearchSettings
    search_mode: Literal["basic", "advanced", "custom"] = "custom"
    task_prompt: Optional[str] = None
    include_title_if_available: bool = False
    include_web_search: bool = False

    @property
    def name(self) -> str:
        """Generate a short name for this configuration"""
        model_name = self.generation_config.model.split("/")[-1]

        if self.search_settings.search_strategy:
            search_type = self.search_settings.search_strategy
        elif self.search_settings.use_hybrid_search:
            search_type = "hybrid"
        else:
            search_type = "semantic"

        return f"{model_name}_{search_type}_{self.search_settings.limit}"

    def to_rag_params(self) -> Dict[str, Any]:
        """Convert the configuration to R2R RAG API parameters"""
        params = {
            "rag_generation_config": self.generation_config.to_dict(),
            "search_settings": self.search_settings.to_dict(),
            "search_mode": self.search_mode,
            "include_title_if_available": self.include_title_if_available,
            "include_web_search": self.include_web_search,
        }

        if self.task_prompt:
            params["task_prompt"] = self.task_prompt

        return params


@dataclass
class MetricsConfig:
    """Configuration for metrics evaluation"""

    llm_model: str
    llm_api_base: str
    embeddings_model: str
    embeddings_api_base: str
    api_key_llm: str = "random_api_key"
    api_key_embeddings: str = "random_api_key"


@dataclass
class TimingConfig:
    """Configuration for timing measurements"""

    enabled: bool = True
    include_steps: List[str] = field(
        default_factory=lambda: ["chunking", "ingestion", "retrieval", "evaluation"]
    )


@dataclass
class ChunkEnrichmentSettings:
    """Settings for chunk enrichment."""

    enable_chunk_enrichment: bool = False
    n_chunks: int = 2
    generation_config: Optional[RAGGenerationConfig] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enable_chunk_enrichment": self.enable_chunk_enrichment,
            "n_chunks": self.n_chunks,
            "generation_config": (
                self.generation_config.to_dict() if self.generation_config else None
            ),
        }

    def __str__(self) -> str:
        return f"enable_chunk_enrichment={self.enable_chunk_enrichment}, n_chunks={self.n_chunks}"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class IngestionConfig:
    """Configuration for ingestion"""

    chunk_enrichment_settings: ChunkEnrichmentSettings = field(
        default_factory=ChunkEnrichmentSettings
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_enrichment": {
                "enable_chunk_enrichment": self.chunk_enrichment_settings.enable_chunk_enrichment,
                "n_chunks": self.chunk_enrichment_settings.n_chunks,
                "generation_config": (
                    self.chunk_enrichment_settings.generation_config.to_dict()
                    if self.chunk_enrichment_settings.generation_config
                    else None
                ),
            }
        }

    def __str__(self) -> str:
        return f"ingestion_config={self.to_dict()}"

    def __repr__(self) -> str:
        return self.__str__()
