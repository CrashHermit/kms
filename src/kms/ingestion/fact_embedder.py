r"""
Fact embedding — one LangGraph node, one batched embedding pass.

The atomic fact pass leaves the ``atomic_facts`` channel carrying only text
and node ids. This stage enriches each fact with its embedding vector — one
batched call over every fact text — so the concept pass can cluster over
fact vectors and the persister can store the vector on the ``:Fact`` node.

A SEPARATE stage rather than part of fact extraction or of graph writing:

* Extraction wants small windows; the embedder wants one big batch (it
  splits at its own batch size). Folding embedding into the fact pass would
  fire tiny per-window requests.
* The concept pass needs the vectors too — computing them here once serves
  both consumers, and the enriched channel is what the persister writes.
* Graph writing is the terminal persister's concern; embedding is a
  semantic enrichment needed before it, and needed even when no Neo4j is
  configured.

When embedding isn't configured the node is a no-op — facts survive without
vectors, and the persister writes ``:Fact`` without the embedding property.
"""

import logging

from kms.core import embeddings, models, state

logger = logging.getLogger(__name__)


async def embed_facts(
    facts: list[models.AtomicFact],
    embedder: embeddings.Embedder,
) -> list[models.AtomicFact]:
    """Embed every fact's text and return the enriched facts.

    Args:
        facts: The atomic facts, in document order.
        embedder: The shared embedder.

    Returns:
        The same facts, each with ``embedding`` set.
    """
    vectors = await embedder.embed([fact.text for fact in facts])
    enriched = [
        models.AtomicFact(
            text=fact.text,
            node_ids=list(fact.node_ids),
            embedding=vector,
        )
        for fact, vector in zip(facts, vectors, strict=True)
    ]
    logger.info('fact embedding: %d fact(s) embedded', len(enriched))
    return enriched


class FactEmbedderNode:
    """Enriches the ``atomic_facts`` channel with embedding vectors.

    Runs after the atomic fact pass and before the persister (and, later,
    the concept pass, which clusters over the same vectors). No-op when
    embedding isn't configured — the persister then writes ``:Fact`` nodes
    without the embedding property.

    Args:
        embedder: The shared embedder.
        embedding_configured: Whether an embedding target is wired. When
            False, ``run`` is a no-op.
    """

    def __init__(
        self,
        embedder: embeddings.Embedder,
        embedding_configured: bool = False,
    ) -> None:
        self._embedder = embedder
        self._embedding_configured = embedding_configured

    async def run(self, state: state.State) -> dict:
        """Embed every atomic fact's text.

        Args:
            state: The pipeline state, holding the atomic facts channel.

        Returns:
            The ``atomic_facts`` channel, facts enriched with embeddings,
            or an empty update when embedding isn't configured or there are
            no facts.
        """
        facts = state.get('atomic_facts', [])
        if not self._embedding_configured or not facts:
            return {}
        return {'atomic_facts': await embed_facts(facts, self._embedder)}
