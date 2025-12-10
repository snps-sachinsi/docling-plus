"""Pipelines package for SynDoc."""

from syndoc.pipelines.base import BasePipeline
from syndoc.pipelines.pdf_pipeline import PdfPipeline

__all__ = [
    "BasePipeline",
    "PdfPipeline",
]
