import os
import asyncio
from typing import List
from r2r import R2RClient
import logging
import colorlog
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from config import RAGConfig
from chunking import ChunkingStrategy, LangChainChunking


"""
Using ThreadPoolExecutor makes blocking I/O operations faster in asyncio by running them in separate threads, 
preventing them from blocking the event loop.
The R2R client uses synchronous HTTP calls, which would normally block the entire asyncio event loop while waiting for responses. 
By running these operations in a ThreadPoolExecutor with loop.run_in_executor(), 
they execute in background threads while the event loop continues handling other tasks.
This allows us to:
- Make multiple API calls concurrently
- Process other tasks while waiting for network responses
- Handle more queries simultaneously without getting blocked
"""


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


class RAGTester:
    def __init__(
        self,
        r2r_url: str = "http://localhost:7272",
    ):
        self.r2r_url = r2r_url
        self.client = R2RClient(r2r_url, timeout=300000)
        self.session = None
        self.executor = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300000)
        )
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        if self.executor:
            self.executor.shutdown(wait=True)

    async def delete_all_documents(self) -> None:
        """Delete all documents from R2R asynchronously"""
        try:
            logger.info("Cleaning up existing documents...")
            max_retries = 100
            for retry in range(max_retries):
                documents = self.client.documents.list(limit=1000)
                doc_count = len(documents.results)

                if doc_count == 0:
                    logger.info("No documents found to delete.")
                    return

                logger.info(f"Deleting {doc_count} documents (attempt {retry+1})")

                tasks = [self.delete_document(doc.id) for doc in documents.results]
                await asyncio.gather(*tasks)

                await asyncio.sleep(10)

                documents = self.client.documents.list(limit=1000)
                remaining = len(documents.results)

                if remaining == 0:
                    logger.info("All documents deleted successfully.")
                    return

                if retry < max_retries - 1:
                    logger.warning(
                        f"Retrying deletion for {remaining} remaining documents..."
                    )
                else:
                    logger.error(
                        f"Failed to delete all documents after {max_retries} attempts. {remaining} documents still remain."
                    )
                    await asyncio.sleep(10)

        except Exception as e:
            logger.error(f"Error during delete_all_documents: {e}")

    async def delete_document(self, doc_id: str) -> None:
        """Delete a single document asynchronously"""
        await self.run_in_executor(self.client.documents.delete, str(doc_id))
        return

    async def ingest_chunks(self, chunks: List[str], batch_size: int = 50) -> None:
        """Ingest chunks into R2R in batches asynchronously"""
        num_chunks = len(chunks)
        logger.info(f"Ingesting {num_chunks} chunks into R2R")

        try:
            await self.run_in_executor(self.client.documents.create, chunks=chunks)
            # logger.info(f"Successfully ingested {num_chunks} chunks")
        except Exception as e:
            logger.error(f"Error ingesting chunks: {e}")

    async def process_rag_queries(
        self, questions: List[str], config: RAGConfig
    ) -> List[str]:
        """Process RAG queries with controlled concurrency"""
        total_questions = len(questions)
        logger.info(f"Processing {total_questions} RAG queries")

        semaphore = asyncio.Semaphore(os.cpu_count())

        responses = [None] * total_questions

        async def process_with_semaphore(i, question):
            async with semaphore:
                try:
                    response = await self.process_query(
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

        failed_count = sum(1 for r in responses if r is None)
        if failed_count > 0:
            logger.warning(
                f"{failed_count}/{total_questions} queries failed and will be skipped in evaluation"
            )

        valid_responses = [r if r is not None else "" for r in responses]

        logger.info(
            f"Completed processing {total_questions - failed_count} successful RAG queries"
        )
        return valid_responses

    async def process_query(
        self, index: int, question: str, config: RAGConfig, total: int
    ) -> str:
        """Process a single RAG query asynchronously with retries"""
        max_retries = 10
        base_delay = 10

        for retry in range(max_retries):
            try:
                response = await self.run_in_executor(
                    self.client.retrieval.rag, query=question, **config.to_rag_params()
                )
                if (index + 1) % 10 == 0 or index == 0 or index == total - 1:
                    logger.info(f"Processed query {index+1}/{total}")
                return response
            except Exception as e:
                if retry < max_retries - 1:
                    delay = base_delay * (2**retry)
                    logger.warning(
                        f"Error processing query {index+1}: {str(e)}. Retrying in {delay} seconds (attempt {retry+1}/{max_retries})..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Failed to process query {index+1} after {max_retries} attempts: {str(e)}"
                    )
                    raise

    async def run_in_executor(self, func, *args, **kwargs):
        """Helper function to run synchronous code in the executor"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, lambda: func(*args, **kwargs))
