"""Enumerations for SynDoc library."""

from enum import Enum


class InputFormat(str, Enum):
    """Supported input document formats."""
    
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    MARKDOWN = "md"
    IMAGE = "image"


class ConversionStatus(str, Enum):
    """Status of document conversion."""
    
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    PENDING = "pending"


class ElementType(str, Enum):
    """Types of document elements detected."""
    
    TEXT = "text"
    TITLE = "title"
    SECTION_HEADER = "section_header"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    FIGURE = "figure"
    CHART = "chart"
    FORMULA = "formula"
    CODE = "code"
    CAPTION = "caption"
    PAGE_HEADER = "page_header"
    PAGE_FOOTER = "page_footer"
    FOOTNOTE = "footnote"


class ProcessingStage(str, Enum):
    """Pipeline processing stages."""
    
    BACKEND_LOADING = "backend_loading"
    PAGE_EXTRACTION = "page_extraction"
    LAYOUT_DETECTION = "layout_detection"
    OCR = "ocr"
    TABLE_STRUCTURE = "table_structure"
    READING_ORDER = "reading_order"
    ENRICHMENT = "enrichment"


class ModelType(str, Enum):
    """Types of models that can be configured."""
    
    LAYOUT_DETECTION = "layout_detection"
    OCR = "ocr"
    TABLE_STRUCTURE = "table_structure"
    READING_ORDER = "reading_order"


class DeviceType(str, Enum):
    """Compute device types."""
    
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon
