"""Live smoke test for the TripletExtractor — exercises the real LM.

Run from the repo root with:
    .venv/bin/python tests/live_triplet_test.py
"""

import asyncio

from kms.core import llm
from kms.ingestion.triplet_extractor import TripletExtractor


SAMPLE_FACTS = [
    # Single relation — should yield 1 triplet
    'The discriminant of $ax^2 + bx + c = 0$ is $b^2 - 4ac$.',

    # Two relations conjoined — should yield 2 triplets
    'The set $\\mathbb{R}$ is uncountable and has cardinality $2^{\\aleph_0}$.',

    # Definition with condition — may yield 1 or 2 triplets
    'A function $f$ is continuous at $c$ if $\\lim_{x\\to c} f(x) = f(c)$.',

    # Instruction-wrapped assertion
    'Prove that every continuous function on $[0,1]$ is bounded.',

    # Simple property
    'The derivative of $\\sin x$ is $\\cos x$.',

    # Fact that asserts something hard to decompose — should yield 0 triplets
    'There exist infinitely many prime numbers.',
]


async def main():
    lm = llm.text_lm()
    extractor = TripletExtractor(language_model=lm)

    for i, fact_text in enumerate(SAMPLE_FACTS):
        print(f'{"="*60}')
        print(f'Fact {i}: {fact_text}')
        print(f'{"-"*60}')
        triplets = await extractor.aforward(fact_text)
        if not triplets:
            print('  (no triplets)')
        for j, t in enumerate(triplets):
            print(f'  Triplet {j}:')
            print(f'    subject:   {t.subject}')
            print(f'    predicate: {t.predicate}')
            print(f'    object:    {t.object}')
        print()

    print(f'{"="*60}')
    print('Done.')


if __name__ == '__main__':
    asyncio.run(main())
