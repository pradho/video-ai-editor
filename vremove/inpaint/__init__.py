from .base import REGISTRY, Inpainter, get, register  # noqa: F401
from . import median, propainter  # noqa: F401  -- import populates REGISTRY

__all__ = ["REGISTRY", "Inpainter", "get", "register"]
