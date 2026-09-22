name = "cabana"

from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("cabana")
except PackageNotFoundError:
    __version__ = "dev"

from .batch_processor import BatchProcessor
from .cabana import Cabana
from .tma import TMAPreprocessor
