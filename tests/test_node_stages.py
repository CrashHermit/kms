"""Post-flatten node stages (problem_refiner, governor) keyed by node id on the flat stream."""

from module.state import ASTNode, NodeType
from module.problem_refiner import ProblemRefinerNode
from module.instruction_governor import InstructionGovernorNode

# A sentinel module keeps the node's dispatch/collect (pure) off the real LLM constructor.
SENTINEL = object()


def _nodes():
    return [
        ASTNode(type=NodeType.HEADER, content="Problems", id=0, seg_index=0),
        ASTNode(type=NodeType.INSTRUCTION, content="For the following, find x", id=1, seg_index=0),
        ASTNode(type=NodeType.PROBLEM, content="y=2x", id=2, seg_index=0),
        ASTNode(type=NodeType.PROBLEM, content="y=3x", id=3, seg_index=1),  # next page
    ]


def test_problem_refiner_dispatch_per_problem_and_collect_by_id():
    nodes = _nodes()
    pr = ProblemRefinerNode(module=SENTINEL)
    sends = pr.dispatch({"nodes": nodes})
    assert [s.arg["node"].id for s in sends] == [2, 3]
    out = pr.collect({"nodes": nodes, "problem_results": [(2, "1"), (3, "2")]})
    assert [n.number for n in out["nodes"] if n.type == NodeType.PROBLEM] == ["1", "2"]


def test_governor_groups_across_page_boundary_as_one_flat_group():
    gv = InstructionGovernorNode(module=SENTINEL)
    sends = gv.dispatch({"nodes": _nodes()})
    assert len(sends) == 1
    assert [m[0] for m in sends[0].arg["members"]] == [2, 3]  # both problems, across the page seam


def test_governor_collect_annotates_by_id_and_drops_distributed_instruction():
    nodes = _nodes()
    gv = InstructionGovernorNode(module=SENTINEL)
    lead = "For the following, find x"
    out = gv.collect({"nodes": nodes, "governance_results": [(2, lead), (3, lead)]})
    assert NodeType.INSTRUCTION not in [n.type for n in out["nodes"]]  # redundant lead removed
    assert all(n.instruction == lead for n in out["nodes"] if n.type == NodeType.PROBLEM)
