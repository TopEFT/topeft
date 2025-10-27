"""Run 2 analysis scripts and workflow helpers."""

try:
    from .workflow import (
        ChannelPlanner,
        ExecutorFactory,
        HistogramPlan,
        HistogramPlanner,
        HistogramTask,
        RunWorkflow,
        normalize_jet_category,
        run_workflow,
    )
except ImportError:  # pragma: no cover - optional workflow helper
    from .workflow import (  # type: ignore[misc]
        ChannelPlanner,
        ExecutorFactory,
        HistogramPlan,
        HistogramPlanner,
        HistogramTask,
        normalize_jet_category,
        run_workflow,
    )
    RunWorkflow = None  # type: ignore[assignment]
from .quickstart import PreparedSamples, prepare_samples, run_quickstart

__all__ = [
    "ChannelPlanner",
    "ExecutorFactory",
    "HistogramPlan",
    "HistogramPlanner",
    "HistogramTask",
    "normalize_jet_category",
    "run_workflow",
    "PreparedSamples",
    "prepare_samples",
    "run_quickstart",
]

if RunWorkflow is not None:
    __all__.append("RunWorkflow")

