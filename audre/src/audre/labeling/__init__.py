"""Conversation labeling: the layer that turns text into analyzable categories.

Labels are *assigned*, never recorded. Every function here therefore stamps a
``label_source`` so that downstream figures can state what assigned them.
"""

from .base import LabelSet, Labeler, Taxonomy
from .rules import RuleLabeler

__all__ = ["LabelSet", "Labeler", "Taxonomy", "RuleLabeler", "LLMLabeler"]


def __getattr__(name: str):
    # LLMLabeler is imported lazily so that `import audre` does not require the
    # optional HTTP dependency.
    if name == "LLMLabeler":
        from .llm import LLMLabeler

        return LLMLabeler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
