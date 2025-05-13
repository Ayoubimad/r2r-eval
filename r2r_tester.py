"""
RAG testing module for the evaluation framework.

This module provides a client for interacting with the R2R service to test RAG
configurations with various parameters and settings.
"""

import os
import asyncio
from typing import List, Optional, Any, Callable
from concurrent.futures import ThreadPoolExecutor
import aiohttp

from r2r import R2RClient
from config import RAGConfig
from utils import create_logger as utils_create_logger

logger = utils_create_logger("rag_tester")


class ThreadPoolExecutorManager:
    """Manager for thread pool executor resources"""

    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize the thread pool executor manager

        Args:
            max_workers: Maximum number of worker threads
        """
        self.max_workers = max_workers or 16
        self.executor = None

    async def __aenter__(self):
        """Context manager entry point"""
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self.executor

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point"""
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None


class AsyncSessionManager:
    """Manager for aiohttp ClientSession resources"""

    def __init__(self, timeout: int = 300000):
        """
        Initialize the session manager

        Args:
            timeout: Request timeout in milliseconds
        """
        self.timeout = timeout
        self.session = None

    async def __aenter__(self):
        """Context manager entry point"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self.session

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point"""
        if self.session:
            await self.session.close()
            self.session = None


class RAGTester:
    """Client for testing RAG configurations using the R2R service"""

    def __init__(
        self,
        r2r_url: str = "http://localhost:7272",
        timeout: int = 300000,
    ):
        """
        Initialize the RAG tester

        Args:
            r2r_url: URL of the R2R service
            timeout: Request timeout in milliseconds
        """
        self.r2r_url = r2r_url
        self.client = R2RClient(r2r_url, timeout=timeout)
        self.session_manager = AsyncSessionManager(timeout=timeout)
        self.executor_manager = ThreadPoolExecutorManager(max_workers=16)
        self.session = None
        self.executor = None

    async def __aenter__(self):
        """Context manager entry point"""
        await self.session_manager.__aenter__()
        self.executor = await self.executor_manager.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point"""
        await self.executor_manager.__aexit__(exc_type, exc_val, exc_tb)
        await self.session_manager.__aexit__(exc_type, exc_val, exc_tb)

    async def run_in_executor(self, func: Callable, *args, **kwargs) -> Any:
        """
        Run a blocking function in an executor to avoid blocking the event loop

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Function result
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, lambda: func(*args, **kwargs))

    async def delete_all_documents(self, max_retries: int = 100) -> None:
        """
        Delete all documents from R2R asynchronously

        Args:
            max_retries: Maximum number of deletion attempts
        """
        try:
            logger.info("Cleaning up existing documents...")

            for retry in range(max_retries):
                documents = await self._list_documents()
                doc_count = len(documents.results)

                if doc_count == 0:
                    logger.info("No documents found to delete.")
                    return

                logger.info(f"Deleting {doc_count} documents (attempt {retry+1})")
                await self._delete_documents_batch(documents.results)

                if await self._is_document_store_empty():
                    logger.info("All documents deleted successfully.")
                    return

                await asyncio.sleep(10)

                if retry == max_retries - 1:
                    remaining = len((await self._list_documents()).results)
                    logger.error(
                        f"Failed to delete all documents after {max_retries} attempts. "
                        f"{remaining} documents still remain."
                    )

        except Exception as e:
            logger.error(f"Error during delete_all_documents: {e}")

    async def _list_documents(self, limit: int = 1000):
        """
        List documents in the R2R store

        Args:
            limit: Maximum number of documents to retrieve

        Returns:
            List of documents
        """
        return await self.run_in_executor(self.client.documents.list, limit=limit)

    async def _is_document_store_empty(self) -> bool:
        """
        Check if the document store is empty

        Returns:
            True if empty, False otherwise
        """
        documents = await self._list_documents(limit=1)
        return len(documents.results) == 0

    async def _delete_documents_batch(self, documents: List[Any]) -> None:
        """
        Delete a batch of documents

        Args:
            documents: List of document objects to delete
        """
        tasks = [self.delete_document(doc.id) for doc in documents]
        await asyncio.gather(*tasks)

    async def delete_document(self, doc_id: str) -> None:
        """
        Delete a single document asynchronously

        Args:
            doc_id: Document ID to delete
        """
        await self.run_in_executor(self.client.documents.delete, str(doc_id))

    async def ingest_chunks(self, chunks: List[str]) -> None:
        """
        Ingest chunks into R2R asynchronously

        Args:
            chunks: List of text chunks to ingest
        """
        num_chunks = len(chunks)
        logger.info(f"Ingesting {num_chunks} chunks into R2R")

        try:
            await self.run_in_executor(self.client.documents.create, chunks=chunks)
        except Exception as e:
            logger.error(f"Error ingesting chunks: {e}")

    async def process_rag_queries(
        self,
        questions: List[str],
        config: RAGConfig,
        max_concurrency: Optional[int] = None,
    ) -> List[str]:
        """
        Process RAG queries with controlled concurrency

        Args:
            questions: List of questions to process
            config: RAG configuration to use
            max_concurrency: Maximum number of concurrent requests

        Returns:
            List of responses for each question
        """
        total_questions = len(questions)
        logger.info(f"Processing {total_questions} RAG queries")

        concurrency = max_concurrency or 16
        semaphore = asyncio.Semaphore(concurrency)

        responses = [None] * total_questions

        async def process_with_semaphore(i, question):
            async with semaphore:
                try:
                    response = await self._process_query_with_retry(
                        i, question, config, total_questions
                    )
                    responses[i] = response
                except Exception as e:
                    logger.error(
                        f"Query {i+1}/{total_questions} failed completely: {str(e)}"
                    )
                    responses[i] = None  # Mark as failed

        tasks = [
            process_with_semaphore(i, question) for i, question in enumerate(questions)
        ]

        await asyncio.gather(*tasks, return_exceptions=True)

        self._log_query_statistics(responses, total_questions)

        valid_responses = [r if r is not None else "" for r in responses]
        return valid_responses

    def _log_query_statistics(self, responses: List[Optional[str]], total: int) -> None:
        """
        Log statistics about query processing

        Args:
            responses: List of responses (may contain None for failed queries)
            total: Total number of queries
        """
        failed_count = sum(1 for r in responses if r is None)
        success_count = total - failed_count

        if failed_count > 0:
            logger.warning(
                f"{failed_count}/{total} queries failed and will be skipped in evaluation"
            )

        logger.info(f"Completed processing {success_count} successful RAG queries")

    async def _process_query_with_retry(
        self, index: int, question: str, config: RAGConfig, total: int
    ) -> str:
        """
        Process a single RAG query asynchronously with retries

        Args:
            index: Query index
            question: Question to process
            config: RAG configuration to use
            total: Total number of queries

        Returns:
            Response text

        Raises:
            Exception: If processing fails after all retries
        """
        max_retries = 10
        base_delay = 10

        for retry in range(max_retries):
            try:
                response = await self.run_in_executor(
                    self.client.retrieval.rag, query=question, **config.to_rag_params()
                )

                if self._should_log_progress(index, total):
                    logger.info(f"Processed query {index+1}/{total}")

                return response
            except Exception as e:
                if retry < max_retries - 1:
                    delay = base_delay * (2**retry)
                    logger.warning(
                        f"Error processing query {index+1}: {str(e)}. "
                        f"Retrying in {delay} seconds (attempt {retry+1}/{max_retries})..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Failed to process query {index+1} after {max_retries} attempts: {str(e)}"
                    )
                    raise

    def _should_log_progress(self, index: int, total: int) -> bool:
        """
        Determine if progress should be logged for the current index

        Args:
            index: Current query index
            total: Total number of queries

        Returns:
            True if progress should be logged, False otherwise
        """
        return (index + 1) % 10 == 0 or index == 0 or index == total - 1
