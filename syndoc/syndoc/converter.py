"""Main document converter for SynDoc."""

import logging
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

from syndoc.backends.base import AbstractDocumentBackend
from syndoc.backends.pdf_backend import PdfBackend
from syndoc.datamodels.config import BackendConfig, ConversionConfig, ModelConfig
from syndoc.datamodels.document import ConversionResult, InputDocument
from syndoc.datamodels.enums import InputFormat
from syndoc.pipelines.base import BasePipeline
from syndoc.pipelines.pdf_pipeline import PdfPipeline

_log = logging.getLogger(__name__)


class DocumentConverter:
    """Main document converter for SynDoc.
    
    Orchestrates the conversion of documents by:
    1. Detecting document format
    2. Creating appropriate backend
    3. Creating appropriate pipeline
    4. Executing conversion
    
    Example:
        >>> converter = DocumentConverter()
        >>> result = converter.convert("document.pdf")
        >>> print(f"Pages: {len(result.pages)}")
    
    Attributes:
        config: Conversion configuration
    """
    
    def __init__(self, config: Optional[ConversionConfig] = None):
        """Initialize document converter.
        
        Args:
            config: Conversion configuration, uses defaults if None
        """
        self.config = config or ConversionConfig()
        
        _log.info("DocumentConverter initialized")
        if self.config.pipeline.do_layout_detection:
            _log.info("Layout detection: ENABLED")
        if self.config.pipeline.do_ocr:
            _log.info("OCR: ENABLED")
        if self.config.pipeline.do_table_structure:
            _log.info("Table structure: ENABLED")
    
    def convert(
        self,
        source: Union[str, Path, BytesIO],
        filename: Optional[str] = None,
    ) -> ConversionResult:
        """Convert a document.
        
        Args:
            source: Path to document or BytesIO stream
            filename: Filename if using stream
            
        Returns:
            Conversion result
            
        Raises:
            ValueError: If document format not supported
            RuntimeError: If conversion fails and raises_on_error=True
        """
        # Convert string to Path
        if isinstance(source, str):
            source = Path(source)
        
        # Detect format
        format = self._detect_format(source, filename)
        _log.info(f"Detected format: {format.value}")
        
        # Create input document
        input_doc = InputDocument.from_path_or_stream(
            source,
            format=format,
            filename=filename,
        )
        
        # Validate file size
        if input_doc.file_size and input_doc.file_size > self.config.max_file_size:
            input_doc.valid = False
            _log.error(
                f"File too large: {input_doc.file_size} bytes "
                f"(max: {self.config.max_file_size})"
            )
        
        if not input_doc.valid:
            return ConversionResult(
                input=input_doc,
                status="failure",
            )
        
        # Create backend
        backend = self._create_backend(input_doc, source)
        
        # Validate backend
        if not backend.is_valid():
            input_doc.valid = False
            _log.error(f"Invalid document: {input_doc.file_path}")
            return ConversionResult(
                input=input_doc,
                status="failure",
            )
        
        # Create pipeline
        pipeline = self._create_pipeline(format)
        
        # Execute conversion
        _log.info(f"Starting conversion: {input_doc.file_path.name}")
        result = pipeline.execute(
            backend=backend,
            raises_on_error=self.config.raises_on_error,
        )
        
        # Cleanup
        backend.unload()
        
        _log.info(
            f"Conversion complete: {result.status.value} "
            f"({result.success_pages} pages, {result.error_count} errors)"
        )
        
        return result
    
    def _detect_format(
        self,
        source: Union[Path, BytesIO],
        filename: Optional[str] = None,
    ) -> InputFormat:
        """Detect document format.
        
        Args:
            source: Document source
            filename: Filename if using stream
            
        Returns:
            Detected format
            
        Raises:
            ValueError: If format cannot be determined
        """
        if isinstance(source, Path):
            extension = source.suffix.lower()
        elif filename:
            extension = Path(filename).suffix.lower()
        else:
            raise ValueError("Cannot determine format without filename")
        
        # Map extensions to formats
        extension_map = {
            ".pdf": InputFormat.PDF,
            ".docx": InputFormat.DOCX,
            ".html": InputFormat.HTML,
            ".htm": InputFormat.HTML,
            ".md": InputFormat.MARKDOWN,
            ".png": InputFormat.IMAGE,
            ".jpg": InputFormat.IMAGE,
            ".jpeg": InputFormat.IMAGE,
        }
        
        format = extension_map.get(extension)
        if format is None:
            raise ValueError(f"Unsupported format: {extension}")
        
        return format
    
    def _create_backend(
        self,
        input_doc: InputDocument,
        source: Union[Path, BytesIO],
    ) -> AbstractDocumentBackend:
        """Create appropriate backend for document.
        
        Args:
            input_doc: Input document
            source: Document source
            
        Returns:
            Document backend
            
        Raises:
            ValueError: If format not supported
        """
        backend_config = BackendConfig()
        
        if input_doc.format == InputFormat.PDF:
            return PdfBackend(input_doc, source, backend_config)
        else:
            raise ValueError(f"No backend for format: {input_doc.format}")
    
    def _create_pipeline(self, format: InputFormat) -> BasePipeline:
        """Create appropriate pipeline for format.
        
        Args:
            format: Document format
            
        Returns:
            Processing pipeline
            
        Raises:
            ValueError: If format not supported
        """
        if format == InputFormat.PDF:
            return PdfPipeline(
                config=self.config.pipeline,
                profiling_enabled=self.config.enable_profiling,
            )
        else:
            raise ValueError(f"No pipeline for format: {format}")
    
    @classmethod
    def from_config_dict(cls, config_dict: dict) -> "DocumentConverter":
        """Create converter from configuration dictionary.
        
        Args:
            config_dict: Configuration as dictionary
            
        Returns:
            DocumentConverter instance
            
        Example:
            >>> config = {
            ...     "pipeline": {
            ...         "do_layout_detection": True,
            ...         "layout_model": {
            ...             "model_type": "detr",
            ...             "device": "cpu",
            ...         }
            ...     }
            ... }
            >>> converter = DocumentConverter.from_config_dict(config)
        """
        config = ConversionConfig(**config_dict)
        return cls(config)
