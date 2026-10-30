"""C1 correction models, kept separate from the frozen Flow-BC policy."""

from .residual import ResidualCorrection

__all__ = ("ResidualCorrection",)
