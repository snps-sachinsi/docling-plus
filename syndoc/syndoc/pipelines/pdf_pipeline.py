"""PDF pipeline implementation."""

import logging
from typing import Optional

from syndoc.backends.base import PaginatedDocumentBackend
from syndoc.backends.pdf_backend import PdfBackend
from syndoc.datamodels.config import PipelineConfig
from syndoc.datamodels.document import (
    BoundingBox,
    ConversionResult,
    DocumentElement,
    ErrorItem,
    Page,
)
from syndoc.datamodels.enums import ElementType, ProcessingStage
from syndoc.models.base import BaseLayoutModel, BaseOCRModel, BaseTableModel
from syndoc.models.factory import ModelRegistry
from syndoc.pipelines.base import BasePipeline

_log = logging.getLogger(__name__)


class PdfPipeline(BasePipeline):
    """Pipeline for processing PDF documents.
    
    Processes PDFs through multiple stages:
    1. Page extraction
    2. Layout detection (optional)
    3. OCR (optional)
    4. Table structure (optional)
    
    Attributes:
        layout_model: Layout detection model
        ocr_model: OCR model
        table_model: Table structure model
    """
    
    def __init__(self, config: PipelineConfig, profiling_enabled: bool = False):
        """Initialize PDF pipeline.
        
        Args:
            config: Pipeline configuration
            profiling_enabled: Enable profiling
        """
        super().__init__(config, profiling_enabled)
        
        # Initialize models based on config
        self.layout_model: Optional[BaseLayoutModel] = None
        self.ocr_model: Optional[BaseOCRModel] = None
        self.table_model: Optional[BaseTableModel] = None
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize models based on configuration."""
        if self.config.do_layout_detection and self.config.layout_model:
            _log.info(
                f"Initializing layout model: {self.config.layout_model.model_type}"
            )
            self.layout_model = ModelRegistry.create_layout_model(
                self.config.layout_model
            )
        
        if self.config.do_ocr and self.config.ocr_model:
            _log.info(f"Initializing OCR model: {self.config.ocr_model.model_type}")
            self.ocr_model = ModelRegistry.create_ocr_model(self.config.ocr_model)
        
        if self.config.do_table_structure and self.config.table_model:
            _log.info(
                f"Initializing table model: {self.config.table_model.model_type}"
            )
            self.table_model = ModelRegistry.create_table_model(
                self.config.table_model
            )
    
    def _process_document(
        self,
        backend: PaginatedDocumentBackend,
        result: ConversionResult,
        raises_on_error: bool,
    ) -> ConversionResult:
        """Process PDF document.
        
        Args:
            backend: PDF backend
            result: Conversion result
            raises_on_error: Whether to raise on errors
            
        Returns:
            Updated result
        """
        if not isinstance(backend, PdfBackend):
            raise TypeError(f"Expected PdfBackend, got {type(backend)}")
        
        # Ensure backend is loaded
        with self._track_stage(ProcessingStage.BACKEND_LOADING.value):
            if backend.pdf_doc is None:
                backend.load()
        
        total_pages = backend.page_count()
        _log.info(f"Processing {total_pages} pages")
        
        # Process each page
        for page_no in range(1, total_pages + 1):
            try:
                page = self._process_page(backend, page_no)
                result.pages.append(page)
            except Exception as e:
                error = ErrorItem(
                    stage=ProcessingStage.PAGE_EXTRACTION.value,
                    message=str(e),
                    exception_type=type(e).__name__,
                    page_no=page_no,
                )
                result.errors.append(error)
                _log.error(f"Error processing page {page_no}: {e}")
                
                if raises_on_error:
                    raise
        
        return result
    
    def _process_page(self, backend: PdfBackend, page_no: int) -> Page:
        """Process a single page.
        
        Args:
            backend: PDF backend
            page_no: Page number
            
        Returns:
            Processed page
        """
        _log.debug(f"Processing page {page_no}")
        
        # Extract page
        with self._track_stage(ProcessingStage.PAGE_EXTRACTION.value):
            page = backend.load_page(page_no)
        
        # Render page image if needed for layout detection
        if self.config.do_layout_detection and self.layout_model:
            with self._track_stage(ProcessingStage.LAYOUT_DETECTION.value):
                page = self._detect_layout(backend, page)
        
        # Extract text if no layout detection
        elif self.config.do_ocr and self.ocr_model:
            with self._track_stage(ProcessingStage.OCR.value):
                page = self._extract_text_ocr(backend, page)
        else:
            # Simple text extraction from PDF
            with self._track_stage(ProcessingStage.PAGE_EXTRACTION.value):
                text = backend.extract_page_text(page_no)
                if text.strip():
                    element = DocumentElement(
                        element_id=f"page_{page_no}_text",
                        element_type=ElementType.TEXT,
                        bbox=BoundingBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0),
                        text=text,
                    )
                    page.elements.append(element)
        
        _log.debug(f"Page {page_no}: {len(page.elements)} elements detected")
        return page
    
    def _detect_layout(self, backend: PdfBackend, page: Page) -> Page:
        """Detect layout elements on page.
        
        Args:
            backend: PDF backend
            page: Page to process
            
        Returns:
            Page with detected elements
        """
        if self.layout_model is None:
            return page
        
        # Render page as image
        page_image = backend.render_page_image(
            page.page_no,
            dpi=self.config.layout_model.custom_params.get("dpi", 144),
        )
        
        # Run layout detection
        predictions = self.layout_model.predict([page_image])
        
        if not predictions or len(predictions) == 0:
            return page
        
        pred = predictions[0]
        boxes = pred.get("boxes", [])
        labels = pred.get("labels", [])
        scores = pred.get("scores", [])
        
        # Convert predictions to elements
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            if score < self.config.layout_model.threshold:
                continue
            
            # Normalize coordinates
            x0, y0, x1, y1 = box
            width, height = page_image.size
            
            element = DocumentElement(
                element_id=f"page_{page.page_no}_elem_{i}",
                element_type=self._map_label_to_element_type(label),
                bbox=BoundingBox(
                    x0=x0 / width,
                    y0=y0 / height,
                    x1=x1 / width,
                    y1=y1 / height,
                ),
                confidence=score,
                metadata={"raw_label": label},
            )
            page.elements.append(element)
        
        return page
    
    def _extract_text_ocr(self, backend: PdfBackend, page: Page) -> Page:
        """Extract text using OCR.
        
        Args:
            backend: PDF backend
            page: Page to process
            
        Returns:
            Page with OCR text
        """
        if self.ocr_model is None:
            return page
        
        # Render page
        page_image = backend.render_page_image(page.page_no)
        
        # Run OCR on full page
        text_results = self.ocr_model.predict(page_image, [[0, 0, 1, 1]])
        
        if text_results and text_results[0]:
            element = DocumentElement(
                element_id=f"page_{page.page_no}_ocr",
                element_type=ElementType.TEXT,
                bbox=BoundingBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0),
                text=text_results[0],
            )
            page.elements.append(element)
        
        return page
    
    def _map_label_to_element_type(self, label: str) -> ElementType:
        """Map model label to element type.
        
        Args:
            label: Model-specific label
            
        Returns:
            ElementType
        """
        # Simple mapping - can be extended
        label_lower = str(label).lower()
        
        if "table" in label_lower:
            return ElementType.TABLE
        elif "figure" in label_lower or "image" in label_lower:
            return ElementType.FIGURE
        elif "title" in label_lower:
            return ElementType.TITLE
        elif "header" in label_lower:
            return ElementType.PAGE_HEADER
        elif "footer" in label_lower:
            return ElementType.PAGE_FOOTER
        elif "caption" in label_lower:
            return ElementType.CAPTION
        else:
            return ElementType.TEXT
