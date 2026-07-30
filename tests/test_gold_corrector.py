"""Integrity of the corrector gold set. Pure file checks — no network, no LLM.

The gold set is hand-written data (data/gold/corrector), so nothing regenerates
it if it drifts. These checks pin the invariants an optimizer run would
otherwise discover the hard way: that every record's files exist, that every
annotated edit really is the difference between the pair, and that a perturbed
record never lands in a different split from the page whose gold it copies.
"""

import json
from pathlib import Path

import pytest

GOLD = Path(__file__).parents[1] / 'data' / 'gold' / 'corrector'
INDEX = json.loads((GOLD / 'index.json').read_text())
RECORDS = INDEX['records']
BY_ID = {record['id']: record for record in RECORDS}


def _text(record: dict, field: str) -> str:
    return (GOLD / record[field]).read_text()


@pytest.mark.parametrize(
    'record', RECORDS, ids=[record['id'] for record in RECORDS]
)
def test_record_files_exist_and_are_non_empty(record):
    for field in ('transcription', 'corrected'):
        assert _text(record, field).strip(), f'{record["id"]}: empty {field}'
    assert (GOLD.parents[2] / record['source_pdf']).exists(), (
        f'{record["id"]}: missing source pdf'
    )


@pytest.mark.parametrize(
    'record', RECORDS, ids=[record['id'] for record in RECORDS]
)
def test_annotated_edits_match_the_pair(record):
    """Each edit's before/after is real, and no edit means an identical pair."""
    transcription = _text(record, 'transcription')
    corrected = _text(record, 'corrected')

    if not record['edits']:
        assert transcription == corrected, (
            f'{record["id"]}: pair differs but no edits are annotated'
        )
        return

    assert transcription != corrected, (
        f'{record["id"]}: edits are annotated but the pair is identical'
    )
    # An order-class edit describes a resequencing rather than a substring
    # swap, so its before/after name the sequence, not literal file content.
    # Applying every other edit must leave a pure permutation of the lines.
    rewritten = transcription
    for edit in record['edits']:
        if edit['class'] == 'order':
            continue
        assert edit['before'] in transcription, (
            f'{record["id"]}: edit["before"] not in the transcription'
        )
        assert edit['after'] in corrected, (
            f'{record["id"]}: edit["after"] not in the corrected text'
        )
        rewritten = rewritten.replace(edit['before'], edit['after'])

    if any(edit['class'] == 'order' for edit in record['edits']):
        assert sorted(rewritten.split('\n')) == sorted(corrected.split('\n')), (
            f'{record["id"]}: an order edit changed content, not just order'
        )
    else:
        assert rewritten == corrected, (
            f'{record["id"]}: the annotated edits do not account for the '
            f'whole difference between the pair'
        )


def test_perturbed_records_track_their_source_page():
    for record in RECORDS:
        if record['kind'] != 'perturbed':
            continue
        base = BY_ID[record['derived_from']]
        assert _text(record, 'corrected') == _text(base, 'corrected'), (
            f'{record["id"]}: gold differs from the page it was derived from'
        )
        # Same split, or a dev answer is visible in a train demo.
        assert record['split'] == base['split'], (
            f'{record["id"]}: split differs from {base["id"]}'
        )
        assert record['page_image'] == base['page_image']


def test_every_book_is_represented_in_both_splits():
    books = {record['book'] for record in RECORDS}
    for split in ('train', 'dev'):
        covered = {
            record['book'] for record in RECORDS if record['split'] == split
        }
        assert covered == books, f'{split} is missing {books - covered}'


def test_ids_are_unique_and_edit_classes_are_known():
    ids = [record['id'] for record in RECORDS]
    assert len(ids) == len(set(ids))

    # The classes named in the corrector's prompt.
    known = {
        'attachment',
        'extent',
        'substitution',
        'polarity',
        'quantity',
        'relation',
        'position',
        'order',
        'presence',
    }
    for record in RECORDS:
        classes = {edit['class'] for edit in record['edits']}
        assert classes <= known, f'{record["id"]}: unknown class in {classes}'
        assert record['edit_classes'] == sorted(classes)
