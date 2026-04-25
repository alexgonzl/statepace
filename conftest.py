"""Root conftest.py — pytest environment setup.

Sets KMP_DUPLICATE_LIB_OK=TRUE before any test collection so that PyTorch
imports do not abort on macOS when libomp.dylib is loaded multiple times.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
