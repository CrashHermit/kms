"""The optimizer's per-signature metrics (the compile step itself needs live LLMs)."""

from types import SimpleNamespace

from module.optimize import attributor_metric, grouper_metric


def test_attributor_metric_is_exact_role_match():
    gold = SimpleNamespace(roles=["statement", "proof"])
    assert attributor_metric(gold, SimpleNamespace(roles=["statement", "proof"]))
    assert not attributor_metric(gold, SimpleNamespace(roles=["statement", "statement"]))
    assert not attributor_metric(gold, SimpleNamespace(roles=["statement"]))  # length mismatch


def test_grouper_metric_is_order_independent_span_set_match():
    span = lambda t, a, b: SimpleNamespace(type=t, start=a, end=b)
    gold = SimpleNamespace(entities=[span("theorem", 0, 2), span("definition", 3, 3)])
    reordered = SimpleNamespace(entities=[span("definition", 3, 3), span("theorem", 0, 2)])
    wrong_bounds = SimpleNamespace(entities=[span("theorem", 0, 3), span("definition", 3, 3)])
    assert grouper_metric(gold, reordered)
    assert not grouper_metric(gold, wrong_bounds)
    assert not grouper_metric(gold, SimpleNamespace(entities=[]))
