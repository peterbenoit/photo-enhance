from importlib.metadata import PackageNotFoundError, version

from photo_enhance.auto_levels import AutoAnalysis, AutoSettings, auto_enhance
from photo_enhance.pipeline import (
    EnhancementError,
    EnhancementOptions,
    EnhancementResult,
    enhance_image,
)

try:
    __version__ = version("photo-enhance")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "AutoAnalysis",
    "AutoSettings",
    "EnhancementError",
    "EnhancementOptions",
    "EnhancementResult",
    "__version__",
    "auto_enhance",
    "enhance_image",
]
