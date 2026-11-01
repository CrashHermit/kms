"""Finalize processed source identities before evidence can be inspected."""

from kms2.core.model.source_processing.admission import SourceAdmissionResult
from kms2.langgraph.source_admission.state import SourceAdmissionState
from kms2.service.source_admission import SourceAdmissionService


class SourceAdmissionFinalizationNode:
    """Make processed media durable and return a staged snapshot."""

    def __init__(self, service: SourceAdmissionService) -> None:
        self._service = service

    async def run(self, state: SourceAdmissionState) -> dict[str, object]:
        preparation = state.preparation
        assert preparation is not None
        snapshot: SourceAdmissionResult = await self._service.finalize(
            preparation, state.processed_pages
        )
        return {'final_snapshot': snapshot}
