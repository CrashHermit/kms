"""Learning card and review node labels and property builders."""

from uuid import NAMESPACE_URL, uuid5

from kms.core import models

CARD_LABEL = 'Card'
REVIEW_LABEL = 'Review'


def card_uuid(target_uuid: str, *, content_key: str) -> str:
    """Returns the deterministic UUID for one component card variant."""
    return uuid5(NAMESPACE_URL, f'{target_uuid}#card#{content_key}').hex


def review_uuid(card_uuid: str, timestamp: str) -> str:
    """Returns the deterministic uuid for a review event."""
    return uuid5(NAMESPACE_URL, f'{card_uuid}#review#{timestamp}').hex


def card_properties(card: models.Card) -> dict:
    """Build the property dict for a Card node."""
    fsrs = card.fsrs
    properties = {
        'uuid': card.uuid,
        'target_uuid': card.target_uuid,
        'target_kind': card.target_kind,
        'prompt': card.prompt,
        'response': card.response,
        'fsrs_stability': fsrs.stability,
        'fsrs_difficulty': fsrs.difficulty,
        'fsrs_due': fsrs.due,
        'fsrs_interval': fsrs.interval,
        'fsrs_reps': fsrs.reps,
        'fsrs_lapses': fsrs.lapses,
        'fsrs_last_review': fsrs.last_review,
        'created_at': card.created_at,
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
