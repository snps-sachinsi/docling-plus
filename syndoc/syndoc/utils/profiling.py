"""Utility functions for profiling and performance tracking."""

import logging
import time
from contextlib import contextmanager
from typing import Dict, Optional

_log = logging.getLogger(__name__)


class TimeRecorder:
    """Context manager for recording execution time.
    
    Usage:
        with TimeRecorder() as timer:
            # do work
            pass
        print(f"Elapsed: {timer.elapsed}s")
    """
    
    def __init__(self, label: Optional[str] = None):
        """Initialize time recorder.
        
        Args:
            label: Optional label for logging
        """
        self.label = label
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def __enter__(self) -> "TimeRecorder":
        """Start timing."""
        self.start_time = time.perf_counter()
        if self.label:
            _log.debug(f"[{self.label}] Started")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing."""
        self.end_time = time.perf_counter()
        if self.label:
            _log.debug(f"[{self.label}] Completed in {self.elapsed:.3f}s")
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds.
        
        Returns:
            Elapsed time, or 0 if not yet stopped
        """
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.perf_counter()
        return end - self.start_time


class ProfilingTracker:
    """Track profiling information across stages.
    
    Usage:
        tracker = ProfilingTracker()
        with tracker.track("stage1"):
            # do work
            pass
        with tracker.track("stage2"):
            # do work
            pass
        print(tracker.get_summary())
    """
    
    def __init__(self):
        """Initialize profiling tracker."""
        self.stage_times: Dict[str, float] = {}
        self.total_start: Optional[float] = None
        self.total_end: Optional[float] = None
    
    @contextmanager
    def track(self, stage: str):
        """Track time for a specific stage.
        
        Args:
            stage: Stage name
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.stage_times[stage] = self.stage_times.get(stage, 0) + elapsed
    
    def start_total(self):
        """Start tracking total time."""
        self.total_start = time.perf_counter()
    
    def end_total(self):
        """End tracking total time."""
        self.total_end = time.perf_counter()
    
    @property
    def total_time(self) -> float:
        """Get total tracked time.
        
        Returns:
            Total time in seconds
        """
        if self.total_start is None:
            return sum(self.stage_times.values())
        end = self.total_end or time.perf_counter()
        return end - self.total_start
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary of all tracked times.
        
        Returns:
            Dictionary of stage names to times
        """
        summary = dict(self.stage_times)
        summary["total"] = self.total_time
        return summary
