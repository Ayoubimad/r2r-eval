"""
Document model module for the RAG evaluation framework.

This module provides a clean, consistent representation of documents throughout
the evaluation pipeline, including content, metadata, and identifiers.
"""

import uuid
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator


class Document(BaseModel):
    """Model for representing a document in the evaluation pipeline.

    Args:
        content: The text content of the document
        id: Unique identifier for the document
        name: User-friendly name of the document
        meta_data: Additional metadata associated with the document
    """

    content: str = Field(description="The text content of the document")
    id: Optional[str] = Field(
        default=None, description="Unique identifier for the document"
    )
    name: Optional[str] = Field(
        default=None, description="User-friendly name of the document"
    )
    meta_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata associated with the document",
    )

    def set_id_if_none(cls, id_value):
        """Generate a unique ID if none is provided"""
        return id_value or str(uuid.uuid4())

    def add_metadata(self, key: str, value: Any) -> None:
        """
        Add a metadata item to the document

        Args:
            key: Metadata key
            value: Metadata value
        """
        self.meta_data[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get a metadata value from the document

        Args:
            key: Metadata key
            default: Default value if key doesn't exist

        Returns:
            Metadata value or default
        """
        return self.meta_data.get(key, default)

    @property
    def word_count(self) -> int:
        """
        Count the number of words in the document content

        Returns:
            Number of words
        """
        return len(self.content.split())

    @property
    def char_count(self) -> int:
        """
        Count the number of characters in the document content

        Returns:
            Number of characters
        """
        return len(self.content)

    def create_child_document(self, content: str, suffix: str = "chunk") -> "Document":
        """
        Create a child document (e.g., a chunk) from this document

        Args:
            content: Content for the child document
            suffix: Suffix to add to the ID/name

        Returns:
            New Document instance
        """
        child_id = None
        if self.id:
            child_id = f"{self.id}_{suffix}"

        child_name = None
        if self.name:
            child_name = f"{self.name}_{suffix}"

        meta_data = self.meta_data.copy()
        meta_data["parent_id"] = self.id
        meta_data["parent_name"] = self.name

        return Document(
            content=content,
            id=child_id,
            name=child_name,
            meta_data=meta_data,
        )
