from uuid import NAMESPACE_URL, uuid5

from kms.graph import nodes

TRIPLET_HUB_LABEL = 'TripletHub'


def triplet_hub_uuid(
    source: str,
    subj_hub_uuid: str,
    pred_hub_uuid: str,
    obj_hub_uuid: str,
) -> str:
    return uuid5(
        NAMESPACE_URL,
        f'{source}#triplet_hub#'
        f'{subj_hub_uuid}#{pred_hub_uuid}#{obj_hub_uuid}',
    ).hex


def triplet_hub_properties(
    source: str,
    subj_hub_uuid: str,
    pred_hub_uuid: str,
    obj_hub_uuid: str,
) -> dict:
    return {
        'uuid': triplet_hub_uuid(
            source, subj_hub_uuid, pred_hub_uuid, obj_hub_uuid
        ),
        'source': nodes.source_uuid(source),
    }


def canonical_subject_pairs(
    groups: list[dict],
) -> list[dict]:
    return [
        {'hub': g['subj_hub'], 'triplet_hub': g['triplet_hub_uuid']}
        for g in groups
    ]


def canonical_predicate_pairs(
    groups: list[dict],
) -> list[dict]:
    return [
        {'hub': g['pred_hub'], 'triplet_hub': g['triplet_hub_uuid']}
        for g in groups
    ]


def canonical_object_pairs(
    groups: list[dict],
) -> list[dict]:
    return [
        {'hub': g['obj_hub'], 'triplet_hub': g['triplet_hub_uuid']}
        for g in groups
    ]


def supported_by_pairs(
    groups: list[dict],
) -> list[dict]:
    pairs: list[dict] = []
    for g in groups:
        for triplet_uuid_val in g['triplet_uuids']:
            pairs.append({
                'triplet_hub': g['triplet_hub_uuid'],
                'triplet': triplet_uuid_val,
            })
    return pairs

