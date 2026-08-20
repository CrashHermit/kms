"""Learning card and review node labels and property builders."""

from uuid import NAMESPACE_URL, uuid5

from kms.core import models

CARD_LABEL = 'Card'
REVIEW_LABEL = 'Review'


def card_uuid(
    hub_uuid: str,
    *,
    content_key: str | None = None,
) -> str:
    """Returns the deterministic uuid for one hub card variant.

    A hub may produce multiple cards in response to different atomic facts,
    so the fact content participates in the identity.
    """
    suffix = 'default'
    if content_key is not None:
        suffix = f'{suffix}#{content_key}'
    return uuid5(NAMESPACE_URL, f'{hub_uuid}#card#{suffix}').hex


def review_uuid(card_uuid: str, timestamp: str) -> str:
    """Returns the deterministic uuid for a review event."""
    return uuid5(NAMESPACE_URL, f'{card_uuid}#review#{timestamp}').hex


def card_properties(card: models.Card) -> dict:
    """Builds the property dict for a Card node."""
    fsrs = card.fsrs
    properties = {
        'uuid': card.uuid,
        'hub_uuid': card.hub_uuid,
        'hub_kind': card.hub_kind,
        'prompt': card.prompt,
        'response': card.response,
        'status': card.status,
        'fsrs_stability': fsrs.stability,
        'fsrs_difficulty': fsrs.difficulty,
        'fsrs_due': fsrs.due,
        'fsrs_interval': fsrs.interval,
        'fsrs_reps': fsrs.reps,
        'fsrs_lapses': fsrs.lapses,
        'fsrs_last_review': fsrs.last_review,
        'created_at': card.created_at,
        'source': card.source,
    }
    return {k: v for k, v in properties.items() if v is not None}


def review_properties(review: models.Review) -> dict:
    """Builds the property dict for a Review node."""
    properties = {
        'uuid': review.uuid,
        'card_uuid': review.card_uuid,
        'rating': review.rating,
        'timestamp': review.timestamp,
        'response_time_ms': review.response_time_ms,
        'confidence': review.confidence,
        'session_id': review.session_id,
        'session_index': review.session_index,
        'fsrs_state_after': review.fsrs_state_after,
        'source': review.source,
    }
    return {k: v for k, v in properties.items() if v is not None}
