"""Judge whether current knowledge addresses the requested understanding."""

from kms.knowledge.state import KnowledgeState
from kms.knowledge.sufficiency.inference import SufficiencyModule
from kms.knowledge.sufficiency.models import (
    SufficiencyJudgment,
    SufficiencyRequest,
)


class KnowledgeSufficiencyNode:
    def __init__(self, module: SufficiencyModule) -> None:
        self._module = module

    async def run(
        self, state: KnowledgeState
    ) -> dict[str, SufficiencyJudgment]:
        judgment = await self._module.acall(
            request=SufficiencyRequest(
                understanding=state['understanding'], context=state['context']
            )
        )
        return {'judgment': judgment}
