import asyncio

from kms.core import llm
from kms.ingestion.triplet_extractor import TripletExtractor

SAMPLE_FACTS = [
    'The discriminant of $ax^2 + bx + c = 0$ is $b^2 - 4ac$.',
    'The set $\\mathbb{R}$ is uncountable and has cardinality $2^{\\aleph_0}$.',
    'A function $f$ is continuous at $c$ if $\\lim_{x\\to c} f(x) = f(c)$.',
    'Prove that every continuous function on $[0,1]$ is bounded.',
    'The derivative of $\\sin x$ is $\\cos x$.',
    'There exist infinitely many prime numbers.',
]


async def main():
    lm = llm.text_lm()
    extractor = TripletExtractor(language_model=lm)

    for i, fact_text in enumerate(SAMPLE_FACTS):
        print(f'{"=" * 60}')
        print(f'Fact {i}: {fact_text}')
        print(f'{"-" * 60}')
        triplets = await extractor.aforward(fact_text)
        if not triplets:
            print('  (no triplets)')
        for j, t in enumerate(triplets):
            print(f'  Triplet {j}:')
            print(f'    subject:   {t.subject}')
            print(f'    predicate: {t.predicate}')
            print(f'    object:    {t.object}')
        print()

    print(f'{"=" * 60}')
    print('Done.')


if __name__ == '__main__':
    asyncio.run(main())

