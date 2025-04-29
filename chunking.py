"""
Document chunking module that provides strategies for splitting documents into smaller chunks.
Includes an LLM-based chunking strategy that uses AI to find natural breakpoints.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any
import re
from document import Document

from langchain_text_splitters import (
    CharacterTextSplitter,
    TokenTextSplitter,
    RecursiveCharacterTextSplitter,
)

from langchain_openai import ChatOpenAI
from chonkie.chunker import SemanticChunker as ChonkieSemanticChunker
from chonkie.chunker import SDPMChunker as ChonkieSDPMChunker
from generic_embeddings import GenericEmbeddings


class ChunkingStrategy(ABC):
    """Base class for document chunking strategies."""

    @abstractmethod
    def chunk(self, document: Document) -> List[Document]:
        """Split a document into chunks according to the strategy.

        Args:
            document: The document to split into chunks

        Returns:
            A list of Document objects representing the chunks
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    async def chunk_async(self, document: Document) -> List[Document]:
        """Split text into chunks asynchronously using the chunking strategy.

        Args:
            document: The document to split into chunks

        Returns:
            A list of Document objects representing the chunks
        """
        raise NotImplementedError("Subclasses must implement this method")

    def clean_text(self, text: str) -> str:
        """Normalize whitespace in text.

        Args:
            text: The text to clean

        Returns:
            Text cleaned of non-ascii characters, base64 images, and normalized whitespace
        """
        # Remove base64 images
        cleaned_text = re.sub(r"!\[.*?\]\(data:image/[^;]*;base64,[^)]*\)", "", text)
        # Replace multiple newlines with a single newline
        cleaned_text = re.sub(r"\n+", "\n", cleaned_text)
        # Replace multiple spaces with a single space
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)
        # Replace multiple tabs with a single tab
        cleaned_text = re.sub(r"\t+", "\t", cleaned_text)
        # Replace multiple carriage returns with a single carriage return
        cleaned_text = re.sub(r"\r+", "\r", cleaned_text)
        # Replace multiple form feeds with a single form feed
        cleaned_text = re.sub(r"\f+", "\f", cleaned_text)
        # Replace multiple vertical tabs with a single vertical tab
        cleaned_text = re.sub(r"\v+", "\v", cleaned_text)
        # Remove non ascii characters
        cleaned_text = re.sub(r"[^\x00-\x7F]+", "", cleaned_text)
        return cleaned_text


class LangChainChunking(ChunkingStrategy):
    """Chunking strategy that uses LangChain text splitters."""

    def __init__(
        self,
        splitter_type: str = "token",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """Initialize the LangChain chunking strategy.

        Args:
            splitter_type: Type of text splitter to use ('character', 'token')
            chunk_size: Target size of each chunk in characters (or tokens for TokenTextSplitter)
            chunk_overlap: Number of characters/tokens to overlap between chunks
            separators: List of separators to use for splitting
            **kwargs: Additional keyword arguments to pass to the text splitter

        Raises:
            ImportError: If LangChain is not installed
            ValueError: If an invalid splitter_type is provided
        """

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.kwargs = kwargs

        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]

        if splitter_type == "character":
            self.splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separator="",
                is_separator_regex=False,
                **kwargs,
            )
        elif splitter_type == "token":
            self.splitter = TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                **kwargs,
            )
        elif splitter_type == "recursive":
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Invalid splitter_type: {splitter_type}. Must be one of: "
                "character, token, recursive"
            )

    def chunk(self, document: Document) -> List[Document]:
        """Split text into chunks using LangChain text splitters.

        Args:
            document: The document to split into chunks

        Returns:
            List of Document objects representing the chunks
        """
        if len(document.content) <= self.chunk_size:
            return [document]

        clean_content = self.clean_text(document.content)

        text_chunks = self.splitter.split_text(clean_content)

        chunks: List[Document] = []
        for i, chunk_content in enumerate(text_chunks, 1):
            meta_data = document.meta_data.copy()
            meta_data["chunk"] = i
            meta_data["chunk_size"] = len(chunk_content)
            meta_data["splitter_type"] = self.splitter.__class__.__name__

            chunk_id = None
            if document.id:
                chunk_id = f"{document.id}_{i}"
            elif document.name:
                chunk_id = f"{document.name}_{i}"

            chunks.append(
                Document(
                    id=chunk_id,
                    name=document.name,
                    meta_data=meta_data,
                    content=chunk_content,
                )
            )

        return chunks

    async def chunk_async(self, document: Document) -> List[Document]:
        """Split text into chunks asynchronously using LangChain text splitters.

        Args:
            document: The document to split into chunks

        Returns:
            List of Document objects representing the chunks
        """
        if len(document.content) <= self.chunk_size:
            return [document]

        clean_content = self.clean_text(document.content)

        text_chunks = self.splitter.split_text(clean_content)

        chunks: List[Document] = []
        for i, chunk_content in enumerate(text_chunks, 1):
            meta_data = document.meta_data.copy()
            meta_data["chunk"] = i
            meta_data["chunk_size"] = len(chunk_content)
            meta_data["splitter_type"] = self.splitter.__class__.__name__

            chunk_id = None
            if document.id:
                chunk_id = f"{document.id}_{i}"
            elif document.name:
                chunk_id = f"{document.name}_{i}"

            chunks.append(
                Document(
                    id=chunk_id,
                    name=document.name,
                    meta_data=meta_data,
                    content=chunk_content,
                )
            )

        return chunks


class AgenticChunking(ChunkingStrategy):
    """Chunking strategy that uses an LLM to determine natural breakpoints in the text"""

    def __init__(self, model: ChatOpenAI, max_chunk_size: int = 32000):
        self.max_chunk_size = max_chunk_size
        self.model = model

    def _prepare_chunk(
        self, document: Document, chunk_text: str, chunk_number: int
    ) -> Document:
        """Create a Document object for a chunk of text"""
        meta_data = document.meta_data.copy() if document.meta_data else {}
        meta_data["chunk"] = chunk_number
        meta_data["chunk_size"] = len(chunk_text)
        meta_data["splitter_type"] = "agentic"

        chunk_id = None
        if document.id:
            chunk_id = f"{document.id}_{chunk_number}"
        elif document.name:
            chunk_id = f"{document.name}_{chunk_number}"

        return Document(
            id=chunk_id,
            name=document.name,
            meta_data=meta_data,
            content=chunk_text,
        )

    def chunk(self, document: Document) -> List[Document]:
        """Split text into chunks using LLM to determine natural breakpoints based on context"""
        if len(document.content) <= self.max_chunk_size:
            return [document]

        chunks: List[Document] = []
        remaining_text = self.clean_text(document.content)
        chunk_number = 1

        while remaining_text:
            prompt = self._create_breakpoint_prompt(remaining_text)
            try:
                break_point = int(self.model.generate(prompt).strip())
            except Exception:
                break_point = self.max_chunk_size

            chunk_text = remaining_text[:break_point].strip()
            chunks.append(self._prepare_chunk(document, chunk_text, chunk_number))
            chunk_number += 1
            remaining_text = remaining_text[break_point:].strip()

            if not remaining_text:
                break

        return chunks

    async def chunk_async(self, document: Document) -> List[Document]:
        """Split text into chunks asynchronously using LLM to determine natural breakpoints"""
        if len(document.content) <= self.max_chunk_size:
            return [document]

        chunks: List[Document] = []
        remaining_text = self.clean_text(document.content)
        chunk_number = 1

        while remaining_text:
            prompt = self._create_breakpoint_prompt(remaining_text)

            try:
                break_point = int((await self.model.generate_async(prompt)).strip())
            except Exception:
                break_point = self.max_chunk_size

            chunk_text = remaining_text[:break_point].strip()
            chunks.append(self._prepare_chunk(document, chunk_text, chunk_number))
            chunk_number += 1
            remaining_text = remaining_text[break_point:].strip()

            if not remaining_text:
                break

        return chunks

    def _create_breakpoint_prompt(self, text: str) -> str:
        """Create a prompt for finding a natural breakpoint in text."""
        return f"""You are an expert in natural language understanding. Your task is to find the most natural breakpoint in the **given text below**, counting characters **starting from the beginning of this text only** (i.e., character 0 is the first character shown).

        A good breakpoint is one that:
        - Ends a sentence, paragraph, or thought
        - Occurs at a natural pause or topic transition
        - Maximizes semantic completeness without exceeding the text

        Return **only the character index (as an integer)** where the text should be split. Do **not** include any explanation, notes, or formatting.

        Here are some example outputs:
        100  
        219  
        320  
        450  
        512  
        634  
        789

        Now analyze the following text and return the best breakpoint index:\n\n{text[:self.max_chunk_size]}
        """


class SemanticChunker(ChunkingStrategy):
    """Chunking strategy that uses semantic search to determine natural breakpoints in the text"""

    def __init__(
        self,
        embedding_model: GenericEmbeddings,
        chunk_size: int = 256,
        threshold: float = 0.5,
    ):
        self.chunk_size = chunk_size
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.chunker = ChonkieSemanticChunker(
            embedding_model=self.embedding_model,
            chunk_size=self.chunk_size,
            threshold=self.threshold,
        )

    def chunk(self, document: Document) -> List[Document]:
        """Split text into chunks using semantic search to determine natural breakpoints"""
        if len(document.content) <= self.chunk_size:
            return [document]

        chunks: List[Document] = []
        text = self.clean_text(document.content)
        chunks = self.chunker.chunk(text)

        return [
            Document(content=chunk.text, meta_data=document.meta_data)
            for chunk in chunks
        ]

    async def chunk_async(self, document: Document) -> List[Document]:
        """Split text into chunks asynchronously using semantic search to determine natural breakpoints"""
        return self.chunk(document)


class SDPMChunker(ChunkingStrategy):
    """Chunking strategy that uses semantic search to determine natural breakpoints in the text"""

    def __init__(
        self,
        embedding_model: GenericEmbeddings,
        chunk_size: int = 256,
        min_chunk_size: int = 8,
        threshold: float = 0.5,
    ):
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.chunker = ChonkieSDPMChunker(
            embedding_model=self.embedding_model,
            chunk_size=self.chunk_size,
            min_chunk_size=self.min_chunk_size,
            threshold=self.threshold,
        )

    def chunk(self, document: Document) -> List[Document]:
        """Split text into chunks using semantic search to determine natural breakpoints"""
        if len(document.content) <= self.chunk_size:
            return [document]

        chunks: List[Document] = []
        text = self.clean_text(document.content)
        chunks = self.chunker.chunk(text)

        return [
            Document(content=chunk.text, meta_data=document.meta_data)
            for chunk in chunks
            if len(chunk.text) > 0
        ]

    async def chunk_async(self, document: Document) -> List[Document]:
        """Split text into chunks asynchronously using semantic search to determine natural breakpoints"""
        return self.chunk(document)
