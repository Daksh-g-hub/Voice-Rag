import time
from typing import Dict, Optional
from contextlib import contextmanager

class PipelineTimer:
    """
    Sub-millisecond latency profiler for tracking each stage of the Voice-RAG pipeline.
    """
    def __init__(self):
        self.metrics: Dict[str, float] = {}
        self._start_times: Dict[str, float] = {}
        self.pipeline_start: float = time.perf_counter()

    @contextmanager
    def measure(self, stage_name: str):
        """Context manager to measure latency of an individual pipeline stage in milliseconds."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.metrics[stage_name] = round(elapsed_ms, 2)

    def start_stage(self, stage_name: str):
        self._start_times[stage_name] = time.perf_counter()

    def end_stage(self, stage_name: str) -> float:
        if stage_name in self._start_times:
            elapsed_ms = (time.perf_counter() - self._start_times[stage_name]) * 1000.0
            self.metrics[stage_name] = round(elapsed_ms, 2)
            del self._start_times[stage_name]
            return self.metrics[stage_name]
        return 0.0

    def get_summary(self) -> Dict[str, float]:
        total_ms = (time.perf_counter() - self.pipeline_start) * 1000.0
        summary = dict(self.metrics)
        summary["total_pipeline_ms"] = round(total_ms, 2)
        return summary
