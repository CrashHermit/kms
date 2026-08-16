import asyncio

from kms import config
from kms.core import content, embeddings, llm, models, walker


def window_content(
    nodes: list[models.ASTNode],
    node_id: int,
    before_budget: int,
    after_budget: int,
) -> content.Content:
    parts: list[content.TextPart | content.ImagePart] = []
    before = walker.content_before(nodes, node_id, before_budget)
    if before:
        parts.append(content.TextPart(text=before))
    node = nodes[node_id]
    if node.content:
        parts.append(content.TextPart(text=node.content))
    after = walker.content_after(nodes, node_id, after_budget)
    if after:
        parts.append(content.TextPart(text=after))
    if node.image_path:
        image = content.load_image(
            node.image_path,
            max_dim=config.get_settings().image.max_dim,
        )
        if image:
            parts.append(content.ImagePart(image=image))
    return content.Content(parts=parts)


def embedding_text(term: str, description: str | None) -> str:
    return f'{term} : {description}' if description else term


async def describe_terms(
    nodes: list[models.ASTNode],
    terms_by_node: dict[int, set[str]],
    enricher,
    before_budget: int,
    after_budget: int,
    max_concurrency: int,
) -> dict[int, dict[str, str | None]]:
    gate = llm.gate(max_concurrency)
    descriptions: dict[int, dict[str, str | None]] = {}

    async def describe(node_id: int, terms: set[str]) -> None:
        if not terms:
            return
        ordered_terms = sorted(terms)
        async with gate:
            results = await enricher.aforward(
                passage=window_content(
                    nodes, node_id, before_budget, after_budget
                ),
                terms=ordered_terms,
            )
        by_term = {item.term: item.description for item in results}
        descriptions[node_id] = {
            term: by_term.get(term) for term in ordered_terms
        }

    await asyncio.gather(
        *(describe(node_id, terms) for node_id, terms in terms_by_node.items())
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
