import asyncio

from kms import config
from kms.core import content, embeddings, llm, models, walker


def window_content(
    nodes: list[models.Node],
    position: int,
    before_budget: int,
    after_budget: int,
) -> content.Content:
    """Builds multimodal content around a target position.

    Preserves document order: before nodes (with images) -> target node
    (text then image) -> after nodes (with images).
    """
    window = walker.marked_window(
        nodes,
        [position],
        backward_budget=before_budget,
        forward_budget=after_budget,
        marker='target',
    )
    # The marked-window primitive preserves source order and image paths;
    # the canonical adapter interleaves each node label with its image.
    return content.labeled_content(window)


def embedding_text(term: str, description: str | None) -> str:
    return f'{term} : {description}' if description else term


async def describe_terms(
    nodes: list[models.Node],
    terms_by_position: dict[int, set[str]],
    enricher,
    before_budget: int,
    after_budget: int,
    max_concurrency: int,
) -> dict[int, dict[str, str | None]]:
    gate = llm.gate(max_concurrency)
    descriptions: dict[int, dict[str, str | None]] = {}

    async def describe(position: int, terms: set[str]) -> None:
        if not terms:
            return
        ordered_terms = sorted(terms)
        passage = window_content(
            nodes, position, before_budget, after_budget
        )

        async def describe_one(term: str) -> tuple[str, str]:
            async with gate:
                results = await enricher.aforward(
                    passage=passage,
                    terms=[term],
                )
            if len(results) != 1:
                raise ValueError(
                    f'position {position} term {term!r} returned '
                    f'{len(results)} descriptions; expected exactly one'
                )
            result = results[0]
            if result.term != term:
                raise ValueError(
                    f'position {position} returned term {result.term!r}; '
                    f'expected exact term {term!r}'
                )
            return term, result.description

        pairs = await asyncio.gather(
            *(describe_one(term) for term in ordered_terms)
        )
        descriptions[position] = dict(pairs)

    await asyncio.gather(
        *(
            describe(position, terms)
            for position, terms in terms_by_position.items()
        )
    )
    return descriptions


async def embed_descriptions(
    descriptions: dict[int, dict[str, str | None]],
) -> dict[int, dict[str, list[float]]]:
    refs: list[tuple[int, str]] = []
    texts: list[content.Content] = []
    for node_id, terms in descriptions.items():
        for term, description in terms.items():
            refs.append((node_id, term))
            texts.append(
                content.Content.from_text(embedding_text(term, description))
            )
    if not texts:
        return {}
    vectors = await embeddings.embedder().embed(texts)
    result: dict[int, dict[str, list[float]]] = {}
    for (node_id, term), vector in zip(refs, vectors, strict=True):
        result.setdefault(node_id, {})[term] = vector
    return result


def configured_concurrency(stage_name: str) -> int:
    stage = getattr(config.get_settings().stages, stage_name)
    return stage.max_concurrent_calls
