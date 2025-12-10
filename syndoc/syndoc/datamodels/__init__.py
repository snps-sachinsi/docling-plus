"""Data models package for SynDoc."""

from syndoc.datamodels.config import (
    BackendConfig,
    ConversionConfig,
    ModelConfig,
    PipelineConfig,
)
from syndoc.datamodels.document import (
    BoundingBox,
    ConversionResult,
    DocumentElement,
    ErrorItem,
    InputDocument,
    Page,
    PageSize,
    ProfilingInfo,
)
from syndoc.datamodels.enums import (
    ConversionStatus,
    DeviceType,
    ElementType,
    InputFormat,
    ModelType,
    ProcessingStage,
)

__all__ = [
    # Config
    "BackendConfig",
    "ConversionConfig",
    "ModelConfig",
    "PipelineConfig",
    # Document
    "BoundingBox",
    "ConversionResult",
    "DocumentElement",
    "ErrorItem",
    "InputDocument",
    "Page",
    "PageSize",
    "ProfilingInfo",
    # Enums
    "ConversionStatus",
    "DeviceType",
    "ElementType",
    "InputFormat",
    "ModelType",
    "ProcessingStage",
]
