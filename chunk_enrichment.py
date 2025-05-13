"""
Chunk enrichment module for enhancing document chunks with context from surrounding chunks.

This module provides a strategy for enriching chunks by incorporating relevant context
from preceding and succeeding chunks to make each chunk more self-contained and informative.
"""

from typing import List, Optional, Any, Callable
import openai
import asyncio
import logging


class ChunkEnrichmentStrategy:
    """Strategy for enriching chunks with context from surrounding chunks."""

    def __init__(
        self,
        n_chunks: int = 2,
        model: str = "ISTA-DASLab/gemma-3-4b-it-GPTQ-4b-128g",
        api_base: Optional[str] = None,
        api_key: Optional[str] = "random-key",
        temperature: float = 0.8,
        max_tokens: int = 4096,
        client: Optional[openai.AsyncOpenAI] = None,
        concurrency_limit: int = 20,
    ):
        """
        Initialize the chunk enrichment strategy.

        Args:
            n_chunks: Number of surrounding chunks to include as context
            model: LLM model to use for enrichment
            api_base: API base URL for OpenAI compatible endpoint
            api_key: API key for OpenAI compatible endpoint
            temperature: Temperature setting for generation
            max_tokens: Maximum tokens for generation
            client: AsyncOpenAI compatible client (if None, will create one from settings)
            concurrency_limit: Maximum number of concurrent requests
        """
        self.n_chunks = n_chunks
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.concurrency_limit = concurrency_limit
        self.semaphore = asyncio.Semaphore(concurrency_limit)

        if client:
            self.client = client
        elif api_base:
            self.client = openai.AsyncOpenAI(
                base_url=api_base,
                api_key=api_key,
                timeout=10000,
            )
        else:
            self.client = None

    def enrich_chunks(self, chunks: List[str]) -> List[str]:
        """
        Synchronous wrapper for enriching chunks.

        Args:
            chunks: List of raw text chunks to enrich

        Returns:
            List of enriched text chunks
        """
        if not chunks or len(chunks) <= 1 or not self.client:
            return chunks

        return asyncio.run(self.enrich_chunks_async(chunks))

    async def enrich_chunks_async(self, chunks: List[str]) -> List[str]:
        """
        Asynchronously enrich chunks with context from surrounding chunks.

        Args:
            chunks: List of raw text chunks to enrich

        Returns:
            List of enriched text chunks
        """
        if not chunks or len(chunks) <= 1 or not self.client:
            return chunks

        import time

        start_time = time.time()

        tasks = []
        for i, chunk in enumerate(chunks):
            start_idx = max(0, i - self.n_chunks)
            preceding = chunks[start_idx:i]

            end_idx = min(len(chunks), i + self.n_chunks + 1)
            succeeding = chunks[i + 1 : end_idx]

            tasks.append(self._enrich_chunk_async(i, chunk, preceding, succeeding))

        results = await asyncio.gather(*tasks)
        enriched_chunks = [chunk for idx, chunk in sorted(results, key=lambda x: x[0])]

        elapsed = time.time() - start_time
        logging.getLogger("chunk_enrichment").info(
            f"Enriched {len(chunks)} chunks in {elapsed:.2f} seconds."
        )
        return enriched_chunks

    async def _enrich_chunk_async(
        self,
        idx: int,
        chunk_content: str,
        preceding_chunks: List[str],
        succeeding_chunks: List[str],
    ) -> tuple[int, str]:
        """
        Asynchronously enrich a single chunk using the LLM.

        Returns the index and the enriched chunk.
        """
        async with self.semaphore:
            if not self.client:
                return idx, chunk_content

            prompt = self._build_prompt(
                chunk_content, preceding_chunks, succeeding_chunks
            )

            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return idx, response.choices[0].message.content.strip()
            except Exception as e:
                logging.getLogger("chunk_enrichment").error(
                    f"Error enriching chunk {idx}: {e}"
                )
                return idx, chunk_content

    def _build_prompt(
        self,
        chunk_content: str,
        preceding_chunks: List[str],
        succeeding_chunks: List[str],
    ) -> str:
        """Build the prompt for chunk enrichment."""
        preceding_text = "\n\n".join(preceding_chunks) if preceding_chunks else ""
        succeeding_text = "\n\n".join(succeeding_chunks) if succeeding_chunks else ""
        chunk_size = len(chunk_content) + 100

        return f"""
            You are a contextual enrichment expert. Your task is to revise the MAIN CHUNK below by naturally incorporating relevant information from the PRECEDING and SUCCEEDING CONTEXTS to improve clarity, flow, and self-containment.

            MAIN CHUNK:
            {chunk_content}

            PRECEDING CONTEXT:
            {preceding_text}

            SUCCEEDING CONTEXT:
            {succeeding_text}

            Guidelines:
            1. Preserve the original meaning and technical detail of the MAIN CHUNK.
            2. Seamlessly integrate only directly relevant context from the surrounding chunks.
            3. Ensure smooth transitions; do not insert disjointed or redundant text.
            4. The result must stand alone, clear and coherent without relying on outside text.
            5. Maintain existing terminology, references, and tone.
            6. Do not introduce new information not present in the provided context.
            7. Do not remove information from the MAIN CHUNK.
            8. **Keep the final enriched chunk under {chunk_size} characters.**
            9. **Return only the enriched chunk. No explanations, comments, or formatting.**

            ENRICHED CHUNK:
    """
