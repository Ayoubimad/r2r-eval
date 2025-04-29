import importlib.util as importutil
import os
import numpy as np
import warnings
from typing import Any, Callable, List, Optional, Union
from chonkie.embeddings import BaseEmbeddings
from chonkie.tokenizer import WordTokenizer

from openai import OpenAI


class GenericEmbeddings(BaseEmbeddings):
    """Generic embeddings implementation compatible with OpenAI API structure."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        batch_size: int = 128,
        embedding_dimension: int = 1536,
        show_warnings: bool = True,
        **kwargs,
    ):
        """Initialize generic embeddings compatible with OpenAI API structure.

        Args:
            model: Name of the embedding model to use
            api_key: API key (if not provided, looks for OPENAI_API_KEY env var)
            base_url: Base URL for API requests (for non-OpenAI providers)
            organization: Optional organization ID for API requests
            max_retries: Maximum number of retries for failed requests
            timeout: Timeout in seconds for API requests
            batch_size: Maximum number of texts to embed in one API call
            embedding_dimension: The dimension of the embeddings (determined by model)
            show_warnings: Whether to show warnings about token usage
            **kwargs: Additional arguments to pass to the client initialization
        """
        super().__init__()

        self.model = model
        self._dimension = embedding_dimension
        self._batch_size = batch_size
        self._show_warnings = show_warnings
        self._tokenizer = WordTokenizer()

        client_kwargs = {
            "api_key": api_key or os.getenv("OPENAI_API_KEY"),
            "timeout": timeout,
            "max_retries": max_retries,
        }

        if organization:
            client_kwargs["organization"] = organization

        if base_url:
            client_kwargs["base_url"] = base_url

        client_kwargs.update(kwargs)

        self.client = OpenAI(**client_kwargs)

        if self.client.api_key is None and not base_url:
            raise ValueError(
                "API key not found. Either pass it as api_key or set OPENAI_API_KEY environment variable."
            )

    def embed(self, text: str) -> np.ndarray:
        """Get embeddings for a single text.

        Args:
            text: The text to embed

        Returns:
            np.ndarray: Embedding vector for the text
        """
        if self._show_warnings:
            token_count = self.count_tokens(text)
            if token_count > 8191:
                warnings.warn(
                    f"Text has {token_count} tokens which may exceed the model's limit. "
                    "It might be truncated."
                )

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
            )

            return self._import_numpy().array(
                response.data[0].embedding, dtype=self._import_numpy().float32
            )
        except Exception as e:
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=[text],
                )
                return self._import_numpy().array(
                    response.data[0].embedding, dtype=self._import_numpy().float32
                )
            except Exception:
                raise e

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for multiple texts using batched API calls.

        Args:
            texts: List of texts to embed

        Returns:
            List[np.ndarray]: List of embedding vectors for each text
        """
        if not texts:
            return []

        all_embeddings = []
        numpy_module = self._import_numpy()

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]

            if self._show_warnings:
                token_counts = [self.count_tokens(text) for text in batch]
                for text, count in zip(batch, token_counts):
                    if count > 8191:
                        warnings.warn(
                            f"Text has {count} tokens which may exceed the model's limit. "
                            "It might be truncated."
                        )

            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                )

                sorted_embeddings = sorted(response.data, key=lambda x: x.index)
                embeddings = [
                    numpy_module.array(e.embedding, dtype=numpy_module.float32)
                    for e in sorted_embeddings
                ]
                all_embeddings.extend(embeddings)

            except Exception as e:
                if len(batch) > 1:
                    warnings.warn(
                        f"Batch embedding failed: {str(e)}. Trying one by one."
                    )
                    individual_embeddings = [self.embed(text) for text in batch]
                    all_embeddings.extend(individual_embeddings)
                else:
                    raise e

        return all_embeddings

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer.

        Args:
            text: The text to count tokens for

        Returns:
            int: Number of tokens in the text
        """
        return len(self._tokenizer.tokenize(text))

    def get_tokenizer_or_token_counter(self) -> Union[Any, Callable[[str], int]]:
        """Return the tokenizer or token counter object.

        Returns:
            Union[Any, Callable[[str], int]]: Tokenizer object or token counter function
        """
        return self._tokenizer

    @property
    def dimension(self) -> int:
        """Return the embedding dimension.

        Returns:
            int: Dimension of the embedding vectors
        """
        return self._dimension

    def _import_numpy(self) -> Any:
        """Import numpy module lazily.

        Returns:
            module: Numpy module
        """
        try:
            import numpy as np

            return np
        except ImportError:
            raise ImportError(
                "numpy is not available. Please install it via `pip install numpy`"
            )

    def is_available(self) -> bool:
        """Check if this embeddings implementation is available.

        Returns:
            bool: True if the embeddings implementation is available
        """
        return (
            importutil.find_spec("openai") is not None
            and importutil.find_spec("numpy") is not None
        )

    def __repr__(self) -> str:
        """Representation of the GenericEmbeddings instance.

        Returns:
            str: String representation of the GenericEmbeddings instance
        """
        base_url_str = (
            f", base_url={self.client.base_url}"
            if hasattr(self.client, "base_url")
            else ""
        )
        return f"GenericEmbeddings(model={self.model}{base_url_str})"
