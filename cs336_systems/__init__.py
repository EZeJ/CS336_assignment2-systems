import importlib.metadata

try:
    __version__ = importlib.metadata.version("cs336-systems")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from .benchmark import (
    load_config,
    create_random_batch,
    benchmark_model,
)

from .tools import (
    beautify_latex_table,
)

__all__ = (
    "benchmark_model",
    "beautify_latex_table",
    "create_random_batch",
    "load_config",
)
