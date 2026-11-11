"""Explicitly capture public pages without ingesting them."""

from datetime import UTC, datetime

from kms.concurrency import run_blocking_work
from kms.source.capture.inspection import PublicURLInspector
from kms.source.capture.models import PDFCandidate
from kms.source.capture.rendering import GotenbergRenderer, _screenshot_pdf
from kms.source.models.inputs import CapturedPDFSource


class SourceCapture:
    """Inspect a URL and render web pages into captured PDF inputs."""

    def __init__(
        self, inspector: PublicURLInspector, renderer: GotenbergRenderer
    ) -> None:
        self._inspector = inspector
        self._renderer = renderer

    async def capture(
        self, url: str, *, title: str | None = None
    ) -> CapturedPDFSource | PDFCandidate:
        """Return a PDF link or a rendered page as a captured PDF input."""
        inspection = await self._inspector.inspect(url)
        if inspection.media_type == 'application/pdf':
            return PDFCandidate(url=inspection.final_url, title=title)
        screenshot = await self._renderer.screenshot(inspection.final_url)
        data = await run_blocking_work(_screenshot_pdf, screenshot)
        return CapturedPDFSource(
            url=inspection.requested_url,
            resolved_url=inspection.final_url,
            title=title,
            retrieved_at=datetime.now(UTC),
            data=data,
        )
