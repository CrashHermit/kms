import dspy
from pydantic import BaseModel


class SeamNodeDTO(BaseModel):
    """Lightweight DSPy boundary model representing a node's content and type."""
    content: str | None = None
    types: list[str] = []


class Signature(dspy.Signature):
    """
    You are an expert technical editor. Two adjacent textbook element runs share a seam —
    the boundary where one run ends and the next begins. Sometimes a single node
    (paragraph, sentence, equation, list item, Caption, etc.) is split across that boundary,
    producing an incomplete tail in the top run and an incomplete head in the bottom run.

    Your job: decide whether the tail node of the top run and the head node of the
    bottom run are two halves of the same interrupted node. If they are, merge them
    into one coherent node. If they are not (i.e. they are already complete, independent
    nodes that happen to sit at the boundary), return None.

    Use the context nodes (the neighbor just inside each run) only to inform your
    judgment — never include their content in the merged output.

    LATEX FORMAT:
    All mathematical notation must use LaTeX format. Use single dollar signs `$ $`
    for inline math and double dollar signs `$$ $$` for block/display math.
    Preserve existing delimiters and math content exactly.
    """

    top_node_context: SeamNodeDTO | None = dspy.InputField(
        description="The node immediately before the tail of the top element run. Read-only context — do not include its content in the output."
    )
    top_bottom_edge_node: SeamNodeDTO = dspy.InputField(
        description="The tail node of the top element run — the candidate for merging."
    )
    bottom_top_edge_node: SeamNodeDTO = dspy.InputField(
        description="The head node of the bottom element run — the other candidate for merging."
    )
    bottom_node_context: SeamNodeDTO | None = dspy.InputField(
        description="The node immediately after the head of the bottom element run. Read-only context — do not include its content in the output."
    )

    node: SeamNodeDTO | None = dspy.OutputField(
        description="The merged result. If the two edge nodes are split halves of the same node, return a single merged node combining their content. If they are already complete independent nodes, return None."
    )


class Module(dspy.Module):
    def __init__(self):
        super().__init__()
        self.merger = dspy.ChainOfThought(Signature)

    async def aforward(
        self,
        top_bottom_edge_node: SeamNodeDTO,
        bottom_top_edge_node: SeamNodeDTO,
        top_node_context: SeamNodeDTO | None = None,
        bottom_node_context: SeamNodeDTO | None = None,
    ):
        result = await self.merger.acall(
            top_node_context=top_node_context,
            top_bottom_edge_node=top_bottom_edge_node,
            bottom_top_edge_node=bottom_top_edge_node,
            bottom_node_context=bottom_node_context,
        )
        return dspy.Prediction(node=result.node)
