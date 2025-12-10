"""Base pipeline classes for SynDoc."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from syndoc.backends.base import AbstractDocumentBackend
from syndoc.datamodels.config import PipelineConfig
from syndoc.datamodels.document import ConversionResult, ErrorItem, Page, ProfilingInfo
from syndoc.datamodels.enums import ConversionStatus, ProcessingStage
from syndoc.utils.profiling import ProfilingTracker

_log = logging.getLogger(__name__)


class BasePipeline(ABC):
    """Abstract base class for document processing pipelines.
    
    A pipeline orchestrates the conversion of documents by:
    1. Loading pages from backend
    2. Processing pages through stages (layout detection, OCR, etc.)
    3. Assembling and enriching results
    
    Attributes:
        config: Pipeline configuration
        profiling_enabled: Whether to track performance
    """
    
    def __init__(self, config: PipelineConfig, profiling_enabled: bool = False):
        """Initialize pipeline.
        
        Args:
            config: Pipeline configuration
            profiling_enabled: Enable profiling
        """
        self.config = config
        self.profiling_enabled = profiling_enabled
        self._profiler = ProfilingTracker() if profiling_enabled else None
    
    def execute(
        self,
        backend: AbstractDocumentBackend,
        raises_on_error: bool = False,
    ) -> ConversionResult:
        """Execute pipeline on document.
        
        Main entry point for document conversion.
        
        Args:
            backend: Document backend
            raises_on_error: Whether to raise on errors
            
        Returns:
            Conversion result
        """
        # Initialize result
        result = ConversionResult(
            input=backend.input_doc,
            status=ConversionStatus.PENDING,
            start_time=datetime.now(),
        )
        
        if self._profiler:
            self._profiler.start_total()
        
        try:
            # Process document
            result = self._process_document(backend, result, raises_on_error)
            
            # Determine final status
            result.status = self._determine_status(result)
            
        except Exception as e:
            result.status = ConversionStatus.FAILURE
            error = ErrorItem(
                stage=ProcessingStage.BACKEND_LOADING.value,
                message=str(e),
                exception_type=type(e).__name__,
            )
            result.errors.append(error)
            
            if raises_on_error:
                raise
            else:
                _log.error(f"Pipeline execution failed: {e}", exc_info=True)
        
        finally:
            result.end_time = datetime.now()
            
            if self._profiler:
                self._profiler.end_total()
                result.profiling = ProfilingInfo(
                    total_time=self._profiler.total_time,
                    stage_times=dict(self._profiler.stage_times),
                    pages_per_second=(
                        len(result.pages) / self._profiler.total_time
                        if self._profiler.total_time > 0
                        else 0.0
                    ),
                )
        
        return result
    
    @abstractmethod
    def _process_document(
        self,
        backend: AbstractDocumentBackend,
        result: ConversionResult,
        raises_on_error: bool,
    ) -> ConversionResult:
        """Process document through pipeline stages.
        
        Args:
            backend: Document backend
            result: Conversion result to populate
            raises_on_error: Whether to raise on errors
            
        Returns:
            Updated conversion result
        """
        pass
    
    def _determine_status(self, result: ConversionResult) -> ConversionStatus:
        """Determine final conversion status.
        
        Args:
            result: Conversion result
            
        Returns:
            Final status
        """
        if len(result.errors) == 0 and len(result.pages) > 0:
            return ConversionStatus.SUCCESS
        elif len(result.pages) > 0:
            return ConversionStatus.PARTIAL_SUCCESS
        else:
            return ConversionStatus.FAILURE
    
    def _track_stage(self, stage: str):
        """Get context manager for tracking stage time.
        
        Args:
            stage: Stage name
            
        Returns:
            Context manager
        """
        if self._profiler:
            return self._profiler.track(stage)
        else:
            from contextlib import nullcontext
            return nullcontext()
