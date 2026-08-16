import asyncio
import base64
import sys
from pathlib import Path

sys.path.insert(0, '.')

import dspy

from kms.core import llm, search
from kms.graph import db

QUERIES = [
    'subgraph',
    'prove that every induced subgraph is a subgraph',
    'a graph that is not a subgraph because it has an extra edge',
]


def _load_image(path: str) -> dspy.Image:
    encoded = base64.b64encode(Path(path).read_bytes()).decode('utf-8')
    return dspy.Image(url=f'data:image/png;base64,{encoded}')


async def main() -> None:
    if not db.is_configured():
        print(
            'Neo4j not configured. Set KMS_DATABASE__URI, '
            'KMS_DATABASE__USERNAME, KMS_DATABASE__PASSWORD.'
        )
        return

    def _session():
        return db.session()

    image_path = 'output/Segments/Segment_0000/Images/Image_000.png'
    image_queries = [
        ['What does this diagram show?', _load_image(image_path)],
    ]

    for query in QUERIES + image_queries:
        print('=' * 72)
        print(f'QUERY: {query}')
        try:
            groups = await search.search(
                query,
                index_name='entity_hub_embedding',
                text_field='description',
                session_factory=_session,
                top_k=5,
                language_model=llm.module_lm('search_judge'),
            )
            for group in groups:
                print(
                    f'--- sub_query: {group.sub_query} '
                    f'({len(group.results)} results)'
                )
                for result in group.results[:3]:
                    name = result.properties.get('canonical_name', '')
                    desc = result.properties.get('description', '')
                    score = result.score
                    relevant = result.judge_relevant
                    print(
                        f'  [{score:.3f}] {name}: {str(desc)[:100]} '
                        f'(judge_relevant={relevant})'
                    )
        except Exception as exc:
            print(f'ERROR: {type(exc).__name__}: {exc}')


if __name__ == '__main__':
    asyncio.run(main())
