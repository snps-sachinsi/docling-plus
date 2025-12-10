"""PDF backend implementation using PyPDFium2."""

import logging
from io import BytesIO
from pathlib import Path
from typing import Optional, Set, Union

import numpy as np
import pypdfium2 as pdfium
from PIL import Image

from syndoc.backends.base import PaginatedDocumentBackend
from syndoc.datamodels.config import BackendConfig
from syndoc.datamodels.document import InputDocument, Page, PageSize
from syndoc.datamodels.enums import InputFormat

_log = logging.getLogger(__name__)


class PdfBackend(PaginatedDocumentBackend):
    """PDF backend using PyPDFium2.
    
    Provides page-level access to PDF documents, including:
    - Page rendering to images
    - Text extraction
    - Page dimensions
    - Metadata extraction
    
    Attributes:
        pdf_doc: PyPDFium2 document object
    """
    
    def __init__(
        self,
        input_doc: InputDocument,
        path_or_stream: Union[Path, BytesIO],
        config: BackendConfig = BackendConfig(),
    ):
        """Initialize PDF backend.
        
        Args:
            input_doc: Input document metadata
            path_or_stream: PDF file path or stream
            config: Backend configuration
        """
        super().__init__(input_doc, path_or_stream, config)
        self.pdf_doc: Optional[pdfium.PdfDocument] = None
    
    def is_valid(self) -> bool:
        """Check if PDF is valid.
        
        Returns:
            True if PDF can be opened
        """
        try:
            if isinstance(self.path_or_stream, Path):
                test_doc = pdfium.PdfDocument(str(self.path_or_stream))
            else:
                test_doc = pdfium.PdfDocument(self.path_or_stream)
            test_doc.close()
            return True
        except Exception as e:
            _log.error(f"Invalid PDF: {e}")
            return False
    
    def load(self) -> None:
        """Load PDF document.
        
        Raises:
            RuntimeError: If PDF cannot be loaded
        """
        try:
            if isinstance(self.path_or_stream, Path):
                self.pdf_doc = pdfium.PdfDocument(str(self.path_or_stream))
            else:
                self.pdf_doc = pdfium.PdfDocument(self.path_or_stream)
            
            # Update page count in input document
            self.input_doc.page_count = len(self.pdf_doc)
            
            _log.info(
                f"Loaded PDF: {self.input_doc.file_path.name} "
                f"({self.input_doc.page_count} pages)"
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to load PDF: {e}") from e
    
    def unload(self) -> None:
        """Close PDF document and free resources."""
        if self.pdf_doc is not None:
            self.pdf_doc.close()
            self.pdf_doc = None
        
        if isinstance(self.path_or_stream, BytesIO):
            self.path_or_stream.close()
    
    def page_count(self) -> int:
        """Get number of pages.
        
        Returns:
            Number of pages
            
        Raises:
            RuntimeError: If document not loaded
        """
        if self.pdf_doc is None:
            raise RuntimeError("PDF not loaded. Call load() first.")
        return len(self.pdf_doc)
    
    def load_page(self, page_no: int) -> Page:
        """Load a PDF page.
        
        Args:
            page_no: Page number (1-indexed)
            
        Returns:
            Page with basic metadata and optional image
            
        Raises:
            ValueError: If page_no out of range
            RuntimeError: If document not loaded
        """
        if self.pdf_doc is None:
            raise RuntimeError("PDF not loaded. Call load() first.")
        
        if page_no < 1 or page_no > len(self.pdf_doc):
            raise ValueError(
                f"Page {page_no} out of range (1-{len(self.pdf_doc)})"
            )
        
        # PyPDFium2 uses 0-indexing
        pdf_page = self.pdf_doc[page_no - 1]
        
        # Get page dimensions
        width = pdf_page.get_width()
        height = pdf_page.get_height()
        
        page = Page(
            page_no=page_no,
            size=PageSize(width=width, height=height),
        )
        
        return page
    
    def render_page_image(
        self,
        page_no: int,
        scale: float = 1.0,
        dpi: Optional[int] = None,
    ) -> Image.Image:
        """Render page as image.
        
        Args:
            page_no: Page number (1-indexed)
            scale: Scale factor (alternative to DPI)
            dpi: DPI for rendering (overrides scale if provided)
            
        Returns:
            PIL Image
            
        Raises:
            ValueError: If page_no out of range
            RuntimeError: If document not loaded
        """
        if self.pdf_doc is None:
            raise RuntimeError("PDF not loaded. Call load() first.")
        
        if page_no < 1 or page_no > len(self.pdf_doc):
            raise ValueError(
                f"Page {page_no} out of range (1-{len(self.pdf_doc)})"
            )
        
        pdf_page = self.pdf_doc[page_no - 1]
        
        # Calculate scale from DPI if provided
        if dpi is not None:
            # 72 points per inch is PDF standard
            scale = dpi / 72.0
        
        # Render page to bitmap
        bitmap = pdf_page.render(
            scale=scale,
            rotation=0,
        )
        
        # Convert to PIL Image
        pil_image = bitmap.to_pil()
        
        return pil_image
    
    def extract_page_text(self, page_no: int) -> str:
        """Extract text from page.
        
        Args:
            page_no: Page number (1-indexed)
            
        Returns:
            Extracted text
            
        Raises:
            ValueError: If page_no out of range
            RuntimeError: If document not loaded
        """
        if self.pdf_doc is None:
            raise RuntimeError("PDF not loaded. Call load() first.")
        
        if page_no < 1 or page_no > len(self.pdf_doc):
            raise ValueError(
                f"Page {page_no} out of range (1-{len(self.pdf_doc)})"
            )
        
        pdf_page = self.pdf_doc[page_no - 1]
        textpage = pdf_page.get_textpage()
        text = textpage.get_text_range()
        
        return text
    
    @classmethod
    def supported_formats(cls) -> Set[InputFormat]:
        """PDF backend supports PDF format.
        
        Returns:
            Set containing InputFormat.PDF
        """
        return {InputFormat.PDF}
