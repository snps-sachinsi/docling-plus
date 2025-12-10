"""Backends package for SynDoc."""

from syndoc.backends.base import (
    AbstractDocumentBackend,
    DeclarativeDocumentBackend,
    PaginatedDocumentBackend,
)
from syndoc.backends.pdf_backend import PdfBackend

__all__ = [
    "AbstractDocumentBackend",
    "DeclarativeDocumentBackend",
    "PaginatedDocumentBackend",
    "PdfBackend",
]
