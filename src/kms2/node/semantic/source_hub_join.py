"""Join independently persisted source-local semantic hub branches."""

from kms2.langgraph.semantic.state import SemanticState


class SourceHubJoinNode:
    """Provide an explicit terminal synchronization point for hub branches."""

    async def run(self, state: SemanticState) -> dict[str, object]:
        """Return no state changes after all typed hub branches complete."""
        return {}


__all__ = ['SourceHubJoinNode']
