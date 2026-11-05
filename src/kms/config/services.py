"""External-service and optional feature configuration."""

from pathlib import Path

from pydantic import BaseModel


class OCRSettings(BaseModel):
    """Mistral OCR service and artifact materialization settings."""

    api_key: str = ''
    model: str = 'mistral-ocr-4-1'
    url: str = 'https://api.mistral.ai/v1/ocr'
    output_dir: str = 'output'
    render_scale: float = 1.0
    block_crop_scale: float = 1.0


class DatabaseSettings(BaseModel):
    """Neo4j connection settings owned by KMS2."""

    uri: str = ''
    username: str = ''
    password: str = ''
    database: str = 'neo4j'


class TrainingSettings(BaseModel):
    """Optional capture settings for DSPy training examples."""

    examples_directory: Path | None = None
