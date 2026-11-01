"""Prepare and look up immutable source snapshots."""

from kms2.core.model.source_processing.admission import OwnedSourceSnapshot
from kms2.langgraph.source_admission.state import SourceAdmissionState
from kms2.service.source_admission import SourceAdmissionService


class SourceAdmissionPreparationNode:
    """Prepare capture files and check deterministic snapshot reuse."""

    def __init__(self, service: SourceAdmissionService) -> None:
        self._service = service

    async def run(self, state: SourceAdmissionState) -> dict[str, object]:
        result = await self._service.prepare(state.user_uuid, state.material)
        if isinstance(result, OwnedSourceSnapshot):
            return {'final_snapshot': result}
        return {'preparation': result}

    def route(self, state: SourceAdmissionState) -> str:
        """Choose reuse or processing after preparation."""
        return 'existing' if state.final_snapshot is not None else 'process'
