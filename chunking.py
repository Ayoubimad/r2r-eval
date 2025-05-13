"""
Document chunking module that provides strategies for splitting documents into smaller chunks.
Includes various chunking strategies using different techniques: fixed-size, semantic, and LLM-based.
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

    def create_chunk_document(
        self,
        parent_document: Document,
        chunk_content: str,
        chunk_number: int,
        strategy_name: str,
    ) -> Document:
        """Create a Document object for a chunk of text

        Args:
            parent_document: Original document being chunked
            chunk_content: Content for this chunk
            chunk_number: Index/number of this chunk
            strategy_name: Name of the chunking strategy

        Returns:
            Document object representing the chunk
        """
        meta_data = (
            parent_document.meta_data.copy() if parent_document.meta_data else {}
        )

        meta_data["chunk"] = chunk_number
        meta_data["chunk_size"] = len(chunk_content)
        meta_data["splitter_type"] = strategy_name

        chunk_id = None
        if parent_document.id:
            chunk_id = f"{parent_document.id}_{chunk_number}"
        elif parent_document.name:
            chunk_id = f"{parent_document.name}_{chunk_number}"

        return Document(
            id=chunk_id,
            name=parent_document.name,
            meta_data=meta_data,
            content=chunk_content,
        )


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
        self.splitter_type = splitter_type

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

        return [
            self.create_chunk_document(
                parent_document=document,
                chunk_content=chunk_content,
                chunk_number=i,
                strategy_name=self.splitter.__class__.__name__,
            )
            for i, chunk_content in enumerate(text_chunks, 1)
        ]

    async def chunk_async(self, document: Document) -> List[Document]:
        """Split text into chunks asynchronously using LangChain text splitters.

        Args:
            document: The document to split into chunks

        Returns:
            List of Document objects representing the chunks
        """
        return self.chunk(document)


class PromptTemplateManager:
    """Manager for generating prompts for LLM-based chunking"""

    @staticmethod
    def create_breakpoint_prompt(text: str, max_chunk_size: int) -> str:
        """
        Create a prompt for finding a natural breakpoint in text.

        Args:
            text: Text to analyze for breakpoints
            max_chunk_size: Maximum chunk size to consider

        Returns:
            Formatted prompt for an LLM
        """
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

        Now analyze the following text and return the best breakpoint index:\n\n{text[:max_chunk_size]}
        """


class AgenticChunking(ChunkingStrategy):
    """Chunking strategy that uses an LLM to determine natural breakpoints in the text"""

    def __init__(self, model: ChatOpenAI, max_chunk_size: int = 32000):
        """
        Initialize the agentic chunking strategy

        Args:
            model: LLM model to use for determining breakpoints
            max_chunk_size: Maximum chunk size in characters
        """
        self.max_chunk_size = max_chunk_size
        self.model = model
        self.prompt_manager = PromptTemplateManager()

    def chunk(self, document: Document) -> List[Document]:
        """
        Split text into chunks using LLM to determine natural breakpoints based on context

        Args:
            document: Document to chunk

        Returns:
            List of document chunks
        """
        if len(document.content) <= self.max_chunk_size:
            return [document]

        chunks: List[Document] = []
        remaining_text = self.clean_text(document.content)
        chunk_number = 1

        while remaining_text:
            breakpoint_index = self._get_breakpoint(remaining_text)

            chunk_text = remaining_text[:breakpoint_index].strip()
            chunks.append(
                self.create_chunk_document(
                    parent_document=document,
                    chunk_content=chunk_text,
                    chunk_number=chunk_number,
                    strategy_name="agentic",
                )
            )

            chunk_number += 1
            remaining_text = remaining_text[breakpoint_index:].strip()

        return chunks

    def _get_breakpoint(self, text: str) -> int:
        """
        Get a natural breakpoint in text using the LLM

        Args:
            text: Text to analyze

        Returns:
            Character index where text should be split
        """
        prompt = self.prompt_manager.create_breakpoint_prompt(text, self.max_chunk_size)

        try:
            break_point = int(self.model.generate(prompt).strip())
            return min(break_point, self.max_chunk_size, len(text))
        except Exception:
            return min(self.max_chunk_size, len(text))

    async def chunk_async(self, document: Document) -> List[Document]:
        """
        Split text into chunks asynchronously using LLM to determine natural breakpoints

        Args:
            document: Document to chunk

        Returns:
            List of document chunks
        """
        if len(document.content) <= self.max_chunk_size:
            return [document]

        chunks: List[Document] = []
        remaining_text = self.clean_text(document.content)
        chunk_number = 1

        while remaining_text:
            breakpoint_index = await self._get_breakpoint_async(remaining_text)

            chunk_text = remaining_text[:breakpoint_index].strip()
            chunks.append(
                self.create_chunk_document(
                    parent_document=document,
                    chunk_content=chunk_text,
                    chunk_number=chunk_number,
                    strategy_name="agentic",
                )
            )

            chunk_number += 1
            remaining_text = remaining_text[breakpoint_index:].strip()

        return chunks

    async def _get_breakpoint_async(self, text: str) -> int:
        """
        Get a natural breakpoint in text using the LLM asynchronously

        Args:
            text: Text to analyze

        Returns:
            Character index where text should be split
        """
        prompt = self.prompt_manager.create_breakpoint_prompt(text, self.max_chunk_size)

        try:
            break_point = int((await self.model.generate_async(prompt)).strip())
            return min(break_point, self.max_chunk_size, len(text))
        except Exception:
            return min(self.max_chunk_size, len(text))


class SemanticChunker(ChunkingStrategy):
    """Chunking strategy that uses semantic search to determine natural breakpoints in the text"""

    def __init__(
        self,
        embedding_model: GenericEmbeddings,
        chunk_size: int = 256,
        threshold: float = 0.5,
    ):
        """
        Initialize the semantic chunking strategy

        Args:
            embedding_model: Embeddings model for semantic comparison
            chunk_size: Target chunk size in words
            threshold: Similarity threshold for chunk boundaries
        """
        self.chunk_size = chunk_size
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.chunker = ChonkieSemanticChunker(
            embedding_model=self.embedding_model,
            chunk_size=self.chunk_size,
            threshold=self.threshold,
        )

    def chunk(self, document: Document) -> List[Document]:
        """
        Split text into chunks using semantic search to determine natural breakpoints

        Args:
            document: Document to chunk

        Returns:
            List of document chunks
        """
        if len(document.content) <= self.chunk_size:
            return [document]

        text = self.clean_text(document.content)
        semantic_chunks = self.chunker.chunk(text)

        return [
            self.create_chunk_document(
                parent_document=document,
                chunk_content=chunk.text,
                chunk_number=i,
                strategy_name="semantic",
            )
            for i, chunk in enumerate(semantic_chunks, 1)
        ]

    async def chunk_async(self, document: Document) -> List[Document]:
        """
        Split text into chunks asynchronously using semantic search

        Args:
            document: Document to chunk

        Returns:
            List of document chunks
        """
        return self.chunk(document)


class SDPMChunker(ChunkingStrategy):
    """
    Chunking strategy that uses Semantic Double Pass Merging to determine
    natural breakpoints in the text
    """

    def __init__(
        self,
        embedding_model: GenericEmbeddings,
        chunk_size: int = 256,
        min_chunk_size: int = 8,
        threshold: float = 0.5,
    ):
        """
        Initialize the SDPM chunking strategy

        Args:
            embedding_model: Embeddings model for semantic comparison
            chunk_size: Target chunk size in words
            min_chunk_size: Minimum chunk size in words
            threshold: Similarity threshold for chunk boundaries
        """
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
        """
        Split text into chunks using SDPM to determine natural breakpoints

        Args:
            document: Document to chunk

        Returns:
            List of document chunks
        """
        if len(document.content) <= self.chunk_size:
            return [document]

        text = self.clean_text(document.content)
        sdpm_chunks = self.chunker.chunk(text)

        return [
            self.create_chunk_document(
                parent_document=document,
                chunk_content=chunk.text,
                chunk_number=i,
                strategy_name="sdpm",
            )
            for i, chunk in enumerate(sdpm_chunks, 1)
            if chunk.text.strip()
        ]

    async def chunk_async(self, document: Document) -> List[Document]:
        """
        Split text into chunks asynchronously using SDPM

        Args:
            document: Document to chunk

        Returns:
            List of document chunks
        """
        return self.chunk(document)
