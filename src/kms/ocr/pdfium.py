"""Synchronization for PDFium operations in OCR and rendering infrastructure."""

import threading

PDFIUM_LOCK = threading.RLock()
