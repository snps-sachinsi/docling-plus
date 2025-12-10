"""Configuration models for SynDoc library."""

from pathlib import Path
from typing import Optional, Tuple, Set

from pydantic import BaseModel, Field, field_validator

from syndoc.datamodels.enums import DeviceType, ModelType


class ModelConfig(BaseModel):
    """Configuration for a specific model.
    
    Attributes:
        model_type: Type identifier for the model (e.g., "detr", "yolo", "tesseract")
        artifact_path: Path to model weights/artifacts
        device: Compute device to use
        threshold: Confidence threshold for predictions
        batch_size: Batch size for inference
        num_threads: Number of threads for CPU inference
        blacklist_classes: Classes to ignore in predictions
        custom_params: Additional model-specific parameters
    """
    
    model_type: str = Field(
        description="Model type identifier"
    )
    artifact_path: Optional[str] = Field(
        None,
        description="Path to model artifacts"
    )
    device: DeviceType = Field(
        default=DeviceType.CPU,
        description="Compute device"
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold"
    )
    batch_size: int = Field(
        default=1,
        ge=1,
        description="Batch size for inference"
    )
    num_threads: int = Field(
        default=4,
        ge=1,
        description="Number of CPU threads"
    )
    blacklist_classes: Set[str] = Field(
        default_factory=set,
        description="Classes to ignore"
    )
    custom_params: dict = Field(
        default_factory=dict,
        description="Model-specific parameters"
    )


class PipelineConfig(BaseModel):
    """Configuration for document processing pipeline.
    
    Attributes:
        do_layout_detection: Enable layout detection
        do_ocr: Enable OCR
        do_table_structure: Enable table structure recognition
        do_reading_order: Enable reading order detection
        layout_model: Configuration for layout detection model
        ocr_model: Configuration for OCR model
        table_model: Configuration for table structure model
        artifacts_path: Base path for all model artifacts
    """
    
    do_layout_detection: bool = Field(
        default=True,
        description="Enable layout detection"
    )
    do_ocr: bool = Field(
        default=False,
        description="Enable OCR"
    )
    do_table_structure: bool = Field(
        default=False,
        description="Enable table structure recognition"
    )
    do_reading_order: bool = Field(
        default=False,
        description="Enable reading order detection"
    )
    
    layout_model: Optional[ModelConfig] = Field(
        default=None,
        description="Layout detection model config"
    )
    ocr_model: Optional[ModelConfig] = Field(
        default=None,
        description="OCR model config"
    )
    table_model: Optional[ModelConfig] = Field(
        default=None,
        description="Table structure model config"
    )
    
    artifacts_path: Optional[Path] = Field(
        None,
        description="Base path for model artifacts"
    )
    
    @field_validator("artifacts_path")
    @classmethod
    def validate_artifacts_path(cls, v: Optional[Path]) -> Optional[Path]:
        """Ensure artifacts path exists if provided."""
        if v is not None:
            v = Path(v).expanduser().resolve()
            if not v.exists():
                raise ValueError(f"Artifacts path does not exist: {v}")
        return v


class ConversionConfig(BaseModel):
    """Main configuration for document conversion.
    
    Attributes:
        pipeline: Pipeline configuration
        max_file_size: Maximum file size in bytes
        max_num_pages: Maximum number of pages to process
        page_range: Tuple of (start_page, end_page), None means no limit
        raises_on_error: Whether to raise exceptions or capture them
        enable_profiling: Enable performance profiling
    """
    
    pipeline: PipelineConfig = Field(
        default_factory=PipelineConfig,
        description="Pipeline configuration"
    )
    
    max_file_size: int = Field(
        default=100_000_000,  # 100 MB
        ge=0,
        description="Maximum file size in bytes"
    )
    max_num_pages: int = Field(
        default=10000,
        ge=1,
        description="Maximum number of pages"
    )
    page_range: Tuple[int, Optional[int]] = Field(
        default=(1, None),
        description="Page range to process (1-indexed)"
    )
    
    raises_on_error: bool = Field(
        default=False,
        description="Raise exceptions on errors"
    )
    enable_profiling: bool = Field(
        default=False,
        description="Enable performance profiling"
    )
    
    @field_validator("page_range")
    @classmethod
    def validate_page_range(
        cls, v: Tuple[int, Optional[int]]
    ) -> Tuple[int, Optional[int]]:
        """Validate page range."""
        start, end = v
        if start < 1:
            raise ValueError("Start page must be >= 1")
        if end is not None and end < start:
            raise ValueError("End page must be >= start page")
        return v


class BackendConfig(BaseModel):
    """Configuration for document backends.
    
    Attributes:
        parse_images: Whether to parse embedded images
        extract_text: Whether to extract text
        dpi: DPI for rendering pages as images
    """
    
    parse_images: bool = Field(
        default=True,
        description="Parse embedded images"
    )
    extract_text: bool = Field(
        default=True,
        description="Extract text from document"
    )
    dpi: int = Field(
        default=72,
        ge=72,
        le=600,
        description="DPI for page rendering"
    )
