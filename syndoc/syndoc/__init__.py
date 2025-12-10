"""SynDoc - Synopsys Document Parser

A scalable and extensible document parsing library for converting PDF documents
into structured representations.

Example:
    >>> from syndoc import DocumentConverter
    >>> converter = DocumentConverter()
    >>> result = converter.convert("document.pdf")
    >>> print(f"Pages: {len(result.pages)}")
"""

import logging

from syndoc.converter import DocumentConverter
from syndoc.datamodels import (
    ConversionConfig,
    ConversionResult,
    ConversionStatus,
    ElementType,
    InputFormat,
    ModelConfig,
    Page,
    PipelineConfig,
)

__version__ = "0.1.0"

__all__ = [
    "DocumentConverter",
    "ConversionConfig",
    "ConversionResult",
    "ConversionStatus",
    "ElementType",
    "InputFormat",
    "ModelConfig",
    "Page",
    "PipelineConfig",
]

# Configure logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
