"""
Optimise the corrector module with MIPROv2.

Student:  qwen3.7-flash  (cheaper, faster)
Teacher:  qwen3.7-plus   (used to bootstrap few-shot demos)
Both via OpenRouter (Alibaba — single provider, no routing needed).

Usage:
    python -m kms.training.optimize_corrector
"""

from difflib import SequenceMatcher

import dspy

from kms.core import llm
from kms.core.llm import OPENROUTER_ENV_KEY
from kms.core.recorder import load_examples
from kms.ingestion.corrector import Corrector


def _openrouter_lm(model: str) -> dspy.LM:
    return dspy.LM(
        f'openrouter/{model}',
        api_key=llm._require_key(OPENROUTER_ENV_KEY, 'sk-or-...'),
        temperature=0.0,
        max_tokens=128000,
    )


def metric(example: dspy.Example, pred, trace=None) -> float:
    pred_text = pred.corrected if hasattr(pred, 'corrected') else str(pred or '')
    return SequenceMatcher(
        None, example.corrected or '', pred_text or ''
    ).ratio()


def main() -> None:
    student_lm = _openrouter_lm('qwen/qwen3.7-flash')
    teacher_lm = _openrouter_lm('qwen/qwen3.7-plus')

    dspy.settings.configure(lm=student_lm)

    trainset = load_examples(
        'corrector', image_fields=frozenset({'page_image'})
    )
    if not trainset:
        raise RuntimeError(
            'No corrector examples found. Run the pipeline with recording '
            'enabled first (see kms.core.recorder).'
        )

    student = Corrector(language_model=student_lm)

    optimizer = dspy.MIPROv2(
        metric=metric,
        prompt_model=teacher_lm,
        task_model=student_lm,
        max_bootstrapped_demos=2,
        max_labeled_demos=2,
        auto='medium',
    )

    optimized = optimizer.compile(
        student=student,
        trainset=trainset,
    )

    import os
    os.makedirs('output/optimized', exist_ok=True)
    optimized.save('output/optimized/corrector.json')


if __name__ == '__main__':
    main()
