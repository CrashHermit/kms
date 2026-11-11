"""Settings for explicitly capturing public web pages."""

from pydantic import BaseModel, ConfigDict, Field


class SourceCaptureSettings(BaseModel):
    """HTTP settings for URL inspection and page rendering."""

    model_config = ConfigDict(extra='forbid')

    gotenberg_url: str = 'http://127.0.0.1:3000'
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_redirects: int = Field(default=5, ge=0)
