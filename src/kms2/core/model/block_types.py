"""Canonical source block types."""

from enum import StrEnum


class BlockType(StrEnum):
    """Structural types assigned to canonical source blocks."""

    TEXT = 'text'
    EQUATION = 'equation'
    PARAGRAPH = 'paragraph'
    MATH = 'math'
    CODE = 'code'
    LIST = 'list'
    TABLE = 'table'
    IMAGE = 'image'
    CAPTION = 'caption'
    HEADER = 'header'
    BIBLIOGRAPHIC = 'bibliographic'
    NOTE = 'note'
    FOOTER = 'footer'
    ASIDE_TEXT = 'aside_text'
    MARKDOWN = 'markdown'
    INSTRUCTION = 'instruction'
