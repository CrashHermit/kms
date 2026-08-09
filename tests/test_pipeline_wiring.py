import ast
import pathlib
import re

MODULE_DIR = pathlib.Path(__file__).resolve().parent.parent / 'src' / 'kms'


def test_every_send_target_and_dispatch_fallback_is_a_registered_node():
    registered = set(
        re.findall(
            r'add_node\([\'"]([a-z_]+)[\'"]',
            (MODULE_DIR / 'pipeline.py').read_text(),
        )
    )
    send_targets, fallbacks = set(), set()
    for path in MODULE_DIR.rglob('*.py'):
        text = path.read_text()
        send_targets |= set(re.findall(r'Send\([\'"]([a-z_]+)[\'"]', text))
        fallbacks |= set(
            re.findall(r'return sends or [\'"]([a-z_]+)[\'"]', text)
        )

    assert registered, 'no add_node calls matched — the pattern is stale'
    assert send_targets, 'no Send targets matched — the pattern is stale'
    assert fallbacks, 'no dispatch fallbacks matched — the pattern is stale'
    assert not (send_targets - registered), (
        f'Send targets with no node: {send_targets - registered}'
    )
    assert not (fallbacks - registered), (
        f'dispatch fallbacks with no node: {fallbacks - registered}'
    )


def test_all_modules_parse():
    for f in MODULE_DIR.rglob('*.py'):
        ast.parse(f.read_text())

