import asyncio
import json
from pathlib import Path

import dspy
from kms.construction import triplet_extractor

from kms.core import llm
from kms.training import fact_dataset, trainer

DATASET = Path('data/training/fact_gold_v5.jsonl')


async def main() -> None:
    examples = fact_dataset.load_examples(DATASET)
    split = round(len(examples) * 0.75)
    trainset, devset = examples[:split], examples[split:]
    evaluator = trainer.Evaluator()
    language_model = dspy.LM(
        'openai/qwen3.5-9b-text',
        api_base='http://127.0.0.1:8080/v1',
        api_key='not-needed',
        temperature=0.0,
        max_tokens=8192,
        cache=True,
    )
    dspy.configure(lm=language_model)
    predictor = dspy.Predict(triplet_extractor._FactSignature)
    predictor.set_lm(language_model)
    baseline = dspy.Evaluate(
        devset=devset,
        metric=evaluator.metric,
        num_threads=1,
        display_progress=False,
    )(predictor)
    optimizer_student = dspy.Predict(triplet_extractor._FactSignature)
    optimizer_student.set_lm(language_model)
    optimizer_teacher = dspy.Predict(triplet_extractor._FactSignature)
    optimizer_teacher.set_lm(language_model)
    with dspy.context(lm=language_model, num_threads=1):
        compiled = evaluator.compile(
            optimizer_student,
            trainset,
            teacher=optimizer_teacher,
            teacher_lm=language_model,
            max_bootstrapped_demos=4,
            max_labeled_demos=8,
            max_rounds=1,
        )
    compiled.set_lm(language_model)
    optimized = dspy.Evaluate(
        devset=devset,
        metric=evaluator.metric,
        num_threads=1,
        display_progress=False,
    )(compiled)
    print(json.dumps({'baseline': str(baseline), 'optimized': str(optimized)}))


if __name__ == '__main__':
    asyncio.run(main())
