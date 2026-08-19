import asyncio

from kms.construction import formatter
from kms.core import llm

SAMPLES = [
    r'Quadratic Formula',
    r'Given \(ax^2 + bx + c = 0\), the roots are x = (-b ± √(b² - 4ac)) / 2a.',
    r'The shirt costs $20, and the total is 3x + 5 = 20.',
    r'Solve x² = 9 and x ≤ 5.',
    r'The function f(x) = x² is the square, and π ≈ 3.14159.',
    r'Area\n====',
    r'Find the area where 2 ≤ x ≤ 5 and y = x².',
]


async def main():
    module = formatter.Formatter(language_model=llm.module_lm('formatter'))
    for index, sample in enumerate(SAMPLES):
        print(f'=== node {index} input ===')
        print(sample)
        output = await module.aforward(node_content=sample)
        print(f'=== node {index} output ===')
        print(output)
    print('\nDone.')


if __name__ == '__main__':
    asyncio.run(main())
