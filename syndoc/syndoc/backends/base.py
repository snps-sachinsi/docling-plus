"""Abstract base classes for document backends."""

from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Set, Union

from syndoc.datamodels.config import BackendConfig
from syndoc.datamodels.document import InputDocument, Page
from syndoc.datamodels.enums import InputFormat

if TYPE_CHECKING:
    from syndoc.datamodels.document import ConversionResult


class AbstractDocumentBackend(ABC):
    """Abstract base class for document backends.
    
    A backend is responsible for:
    - Validating document format
    - Loading document content
    - Providing page-level access
    - Extracting metadata
    
    Attributes:
        input_doc: Input document metadata
        path_or_stream: File path or stream
        config: Backend configuration
    """
    
    def __init__(
        self,
        input_doc: InputDocument,
        path_or_stream: Union[Path, BytesIO],
        config: BackendConfig = BackendConfig(),
    ):
        """Initialize backend.
        
        Args:
            input_doc: Input document metadata
            path_or_stream: Document file path or stream
            config: Backend configuration
        """
        self.input_doc = input_doc
        self.path_or_stream = path_or_stream
        self.config = config
        
        # Validate format
        if input_doc.format not in self.supported_formats():
            raise ValueError(
                f"Format {input_doc.format} not supported by {self.__class__.__name__}. "
                f"Supported formats: {self.supported_formats()}"
            )
    
    @abstractmethod
    def is_valid(self) -> bool:
        """Check if document is valid and can be processed.
        
        Returns:
            True if document is valid
        """
        pass
    
    @abstractmethod
    def load(self) -> None:
        """Load the document into memory.
        
        Should be called before accessing pages.
        
        Raises:
            RuntimeError: If document cannot be loaded
        """
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """Unload document and free resources."""
        pass
    
    @classmethod
    @abstractmethod
    def supported_formats(cls) -> Set[InputFormat]:
        """Get supported input formats.
        
        Returns:
            Set of supported formats
        """
        pass
    
    @classmethod
    @abstractmethod
    def supports_pagination(cls) -> bool:
        """Check if backend supports pagination.
        
        Returns:
            True if backend can load individual pages
        """
        pass


class PaginatedDocumentBackend(AbstractDocumentBackend):
    """Base class for backends with page-level access.
    
    Extends AbstractDocumentBackend with pagination support.
    """
    
    @abstractmethod
    def page_count(self) -> int:
        """Get total number of pages.
        
        Returns:
            Number of pages in document
        """
        pass
    
    @abstractmethod
    def load_page(self, page_no: int) -> Page:
        """Load a specific page.
        
        Args:
            page_no: Page number (1-indexed)
            
        Returns:
            Page object with basic metadata
            
        Raises:
            ValueError: If page_no is out of range
            RuntimeError: If page cannot be loaded
        """
        pass
    
    @classmethod
    def supports_pagination(cls) -> bool:
        """Paginated backends support pagination."""
        return True


class DeclarativeDocumentBackend(AbstractDocumentBackend):
    """Base class for declarative backends.
    
    Declarative backends can convert entire documents in one step
    without requiring a pipeline (e.g., structured formats like HTML, Markdown).
    """
    
    @abstractmethod
    def convert(self) -> "ConversionResult":
        """Convert entire document.
        
        Returns:
            Complete conversion result
        """
        pass
    
    @classmethod
    def supports_pagination(cls) -> bool:
        """Declarative backends typically don't support pagination."""
        return False
