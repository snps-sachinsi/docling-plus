"""Document data models for SynDoc library."""

import hashlib
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path, PurePath
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from syndoc.datamodels.enums import ConversionStatus, ElementType, InputFormat


class BoundingBox(BaseModel):
    """Bounding box for document elements.
    
    Coordinates are normalized to [0, 1] relative to page dimensions.
    """
    
    x0: float = Field(ge=0.0, le=1.0, description="Left x coordinate")
    y0: float = Field(ge=0.0, le=1.0, description="Top y coordinate")
    x1: float = Field(ge=0.0, le=1.0, description="Right x coordinate")
    y1: float = Field(ge=0.0, le=1.0, description="Bottom y coordinate")
    
    @property
    def width(self) -> float:
        """Width of bounding box."""
        return self.x1 - self.x0
    
    @property
    def height(self) -> float:
        """Height of bounding box."""
        return self.y1 - self.y0
    
    @property
    def area(self) -> float:
        """Area of bounding box."""
        return self.width * self.height


class DocumentElement(BaseModel):
    """A single element in a document (text, table, figure, etc.).
    
    Attributes:
        element_id: Unique identifier for the element
        element_type: Type of element
        bbox: Bounding box coordinates
        text: Text content (if applicable)
        confidence: Confidence score for detection
        metadata: Additional element-specific metadata
    """
    
    element_id: str = Field(description="Unique element ID")
    element_type: ElementType = Field(description="Element type")
    bbox: BoundingBox = Field(description="Bounding box")
    text: Optional[str] = Field(None, description="Text content")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class PageSize(BaseModel):
    """Page dimensions."""
    
    width: float = Field(gt=0, description="Page width in points")
    height: float = Field(gt=0, description="Page height in points")


class Page(BaseModel):
    """A single page in a document.
    
    Attributes:
        page_no: Page number (1-indexed)
        size: Page dimensions
        elements: List of detected elements on the page
        image: Optional page image as base64 or path
        processing_time: Time taken to process this page
    """
    
    page_no: int = Field(ge=1, description="Page number (1-indexed)")
    size: PageSize = Field(description="Page dimensions")
    elements: List[DocumentElement] = Field(
        default_factory=list,
        description="Detected elements"
    )
    image: Optional[str] = Field(
        None,
        description="Page image (base64 or path)"
    )
    processing_time: float = Field(
        default=0.0,
        ge=0.0,
        description="Processing time in seconds"
    )
    
    def __len__(self) -> int:
        """Number of elements on the page."""
        return len(self.elements)


class ErrorItem(BaseModel):
    """Error information during conversion.
    
    Attributes:
        stage: Processing stage where error occurred
        message: Error message
        exception_type: Type of exception
        page_no: Page number where error occurred (if applicable)
    """
    
    stage: str = Field(description="Processing stage")
    message: str = Field(description="Error message")
    exception_type: str = Field(description="Exception type")
    page_no: Optional[int] = Field(
        None,
        description="Page number (if applicable)"
    )


class ProfilingInfo(BaseModel):
    """Performance profiling information.
    
    Attributes:
        total_time: Total conversion time
        stage_times: Time spent in each processing stage
        pages_per_second: Processing throughput
    """
    
    total_time: float = Field(ge=0.0, description="Total time in seconds")
    stage_times: Dict[str, float] = Field(
        default_factory=dict,
        description="Time per stage"
    )
    pages_per_second: Optional[float] = Field(
        None,
        ge=0.0,
        description="Processing throughput"
    )


class InputDocument(BaseModel):
    """Input document metadata.
    
    Attributes:
        file_path: Path to the document file
        document_hash: Hash of document content
        format: Document format
        file_size: Size in bytes
        page_count: Total number of pages
        valid: Whether document is valid and can be processed
    """
    
    file_path: PurePath = Field(description="Document path")
    document_hash: str = Field(description="Document hash")
    format: InputFormat = Field(description="Document format")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    page_count: int = Field(default=0, ge=0, description="Number of pages")
    valid: bool = Field(default=True, description="Document validity")
    
    @classmethod
    def from_path_or_stream(
        cls,
        path_or_stream: Union[Path, BytesIO],
        format: InputFormat,
        filename: Optional[str] = None,
    ) -> "InputDocument":
        """Create InputDocument from file path or stream.
        
        Args:
            path_or_stream: File path or BytesIO stream
            format: Document format
            filename: Filename if using stream
            
        Returns:
            InputDocument instance
        """
        if isinstance(path_or_stream, Path):
            file_path = path_or_stream
            file_size = path_or_stream.stat().st_size
            
            # Compute hash
            hasher = hashlib.sha256()
            with open(path_or_stream, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            document_hash = hasher.hexdigest()
            
        elif isinstance(path_or_stream, BytesIO):
            if filename is None:
                raise ValueError("filename required when using BytesIO")
            file_path = PurePath(filename)
            file_size = path_or_stream.getbuffer().nbytes
            
            # Compute hash
            hasher = hashlib.sha256()
            path_or_stream.seek(0)
            hasher.update(path_or_stream.read())
            path_or_stream.seek(0)
            document_hash = hasher.hexdigest()
        else:
            raise TypeError(f"Unsupported type: {type(path_or_stream)}")
        
        return cls(
            file_path=file_path,
            document_hash=document_hash,
            format=format,
            file_size=file_size,
        )


class ConversionResult(BaseModel):
    """Result of document conversion.
    
    Attributes:
        input: Input document metadata
        status: Conversion status
        pages: List of processed pages
        errors: List of errors encountered
        profiling: Performance profiling information
        metadata: Additional conversion metadata
        start_time: Conversion start time
        end_time: Conversion end time
    """
    
    input: InputDocument = Field(description="Input document")
    status: ConversionStatus = Field(
        default=ConversionStatus.PENDING,
        description="Conversion status"
    )
    pages: List[Page] = Field(
        default_factory=list,
        description="Processed pages"
    )
    errors: List[ErrorItem] = Field(
        default_factory=list,
        description="Errors encountered"
    )
    profiling: Optional[ProfilingInfo] = Field(
        None,
        description="Profiling information"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Conversion metadata"
    )
    start_time: Optional[datetime] = Field(
        None,
        description="Start time"
    )
    end_time: Optional[datetime] = Field(
        None,
        description="End time"
    )
    
    @property
    def duration(self) -> Optional[float]:
        """Total conversion duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def success_pages(self) -> int:
        """Number of successfully processed pages."""
        return len(self.pages)
    
    @property
    def error_count(self) -> int:
        """Number of errors."""
        return len(self.errors)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return self.model_dump_json(indent=2)
