"""Shared synchronization for the non-thread-safe PDFium runtime."""

import threading

PDFIUM_LOCK = threading.RLock()
