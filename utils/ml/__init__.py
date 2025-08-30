"""Machine learning utilities package.

This module lazily exposes the public helpers defined in
``utils.ml.random_forest`` so that importing :mod:`utils.ml` does not incur the
heavy cost (or potential import-time errors) of loading scikit-learn
dependencies unless the functions are actually used.
"""

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "train_model",
    "train_over25_model",
    "save_model",
    "predict_outcome",
    "construct_features_for_match",
    "predict_proba",
    "load_model",
    "load_over25_model",
    "predict_over25_proba",
]

if TYPE_CHECKING:  # pragma: no cover - only for static type checkers
    from .random_forest import (  # noqa: F401
        train_model,
        train_over25_model,
        save_model,
        predict_outcome,
        construct_features_for_match,
        predict_proba,
        load_model,
        load_over25_model,
        predict_over25_proba,
    )


def __getattr__(name: str):  # pragma: no cover - simple delegation
    if name in __all__:
        module = import_module("utils.ml.random_forest")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name}")
