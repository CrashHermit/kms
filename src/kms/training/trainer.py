"""Generic DSPy BootstrapFewShot training support."""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from typing import Protocol

import dspy

from kms.core import module


@dataclass(frozen=True)
class ComparisonResult:
    """Percentage character comparison between expected and predicted values."""

    score: float
    exact_match: bool
    expected: str
    predicted: str
    diff: str


@dataclass(frozen=True)
class JudgeResult:
    """Structured semantic assessment of one prediction."""

    score: float
    passed: bool
    feedback: str


class OutputJudge(Protocol):
    """Optionally semantically assess one prediction."""

    def judge(
        self,
        *,
        input_value: object,
        expected: object,
        predicted: object,
        comparison: ComparisonResult,
        rubric: str = '',
    ) -> JudgeResult:
        """Return a semantic judgment for one prediction."""


class JudgeSignature(dspy.Signature):
    """Assess a prediction against a reference output and task rubric."""

    input_value: str = dspy.InputField()
    expected_output: str = dspy.InputField()
    predicted_output: str = dspy.InputField()
    deterministic_score: float = dspy.InputField()
    difference: str = dspy.InputField()
    rubric: str = dspy.InputField()
    score: float = dspy.OutputField()
    passed: bool = dspy.OutputField()
    feedback: str = dspy.OutputField()


class GenericComparator:
    """Compare arbitrary values through percentage character similarity."""

    def compare(self, expected: object, predicted: object) -> ComparisonResult:
        """Return exactness, percentage similarity, and a unified diff."""
        expected_text = _canonical_text(expected)
        predicted_text = _canonical_text(predicted)
        matcher = difflib.SequenceMatcher(None, expected_text, predicted_text)
        diff = ''.join(
            difflib.unified_diff(
                expected_text.splitlines(keepends=True),
                predicted_text.splitlines(keepends=True),
                fromfile='expected',
                tofile='predicted',
            )
        )
        return ComparisonResult(
            score=matcher.ratio() * 100,
            exact_match=expected_text == predicted_text,
            expected=expected_text,
            predicted=predicted_text,
            diff=diff,
        )


class SimpleJudge(module.Module):
    """Use a DSPy predictor to assess one generic program prediction."""

    signature = JudgeSignature
    record_name = 'training_judge'

    def __init__(self, language_model: dspy.LM) -> None:
        super().__init__(language_model)

    def encode(
        self,
        *,
        input_value: object,
        expected: object,
        predicted: object,
        comparison: ComparisonResult,
        rubric: str = '',
    ) -> dict[str, object]:
        """Serialize comparison inputs for the judge signature."""
        return {
            'input_value': _canonical_text(input_value),
            'expected_output': comparison.expected,
            'predicted_output': comparison.predicted,
            'deterministic_score': comparison.score,
            'difference': comparison.diff,
            'rubric': rubric,
        }

    def decode(
        self, prediction: dspy.Prediction, **inputs: object
    ) -> JudgeResult:
        """Validate the judge's structured result."""
        return JudgeResult(
            score=module.require_number(
                prediction.score, 'score', minimum=0, maximum=1
            ),
            passed=module.require_bool(prediction.passed, 'passed'),
            feedback=module.require_text(prediction.feedback, 'feedback'),
        )


class Evaluator:
    """Expose a character-difference metric for BootstrapFewShot."""

    def __init__(
        self,
        comparator: GenericComparator | None = None,
        judge: OutputJudge | None = None,
        *,
        deterministic_weight: float = 1.0,
        rubric: str = '',
    ) -> None:
        if not 0 <= deterministic_weight <= 1:
            raise ValueError('deterministic_weight must be between 0 and 1')
        self.comparator = comparator or GenericComparator()
        self.judge = judge
        self.deterministic_weight = deterministic_weight
        self.rubric = rubric

    def metric(
        self,
        example: dspy.Example,
        prediction: object,
        trace: object | None = None,
    ) -> float:
        """Return a 0–100 percentage similarity for BootstrapFewShot."""
        inputs = {name: getattr(example, name) for name in example.inputs()}
        output_fields = set(example.toDict()) - set(inputs)
        if len(output_fields) != 1:
            raise ValueError(
                'BootstrapFewShot metric requires exactly one output field'
            )
        expected = getattr(example, next(iter(output_fields)))
        predicted = _prediction_value(prediction)
        comparison = self.comparator.compare(expected, predicted)
        if self.judge is None or self.deterministic_weight == 1:
            return comparison.score
        judgment = self.judge.judge(
            input_value=inputs,
            expected=expected,
            predicted=predicted,
            comparison=comparison,
            rubric=self.rubric,
        )
        return (
            self.deterministic_weight * comparison.score
            + (1 - self.deterministic_weight) * judgment.score
        )

    def compile(
        self,
        student: dspy.Module,
        trainset: list[dspy.Example],
        *,
        teacher: dspy.Module | None = None,
        teacher_lm: dspy.LM | None = None,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 16,
        max_rounds: int = 1,
    ) -> dspy.Module:
        """Compile a student with DSPy's BootstrapFewShot optimizer."""
        optimizer_kwargs: dict[str, object] = {
            'metric': self.metric,
            'max_bootstrapped_demos': max_bootstrapped_demos,
            'max_labeled_demos': max_labeled_demos,
            'max_rounds': max_rounds,
        }
        if teacher_lm is not None:
            teacher = teacher or student.deepcopy()
            teacher.set_lm(teacher_lm)
        optimizer = dspy.BootstrapFewShot(**optimizer_kwargs)
        return optimizer.compile(student, teacher=teacher, trainset=trainset)


def configured_judge() -> SimpleJudge:
    """Construct the configured Nemotron-backed training judge."""
    from kms.core import llm

    return SimpleJudge(llm.module_lm('training_judge'))


def _prediction_value(prediction: object) -> object:
    """Extract one output from a DSPy prediction or return a plain value."""
    if isinstance(prediction, dspy.Prediction):
        fields = dict(prediction)
        if len(fields) != 1:
            raise ValueError(
                'BootstrapFewShot metric requires one predicted output field'
            )
        return next(iter(fields.values()))
    return prediction


def _canonical_text(value: object) -> str:
    """Return stable text for arbitrary output values."""
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, default=str
        )
    except (TypeError, ValueError):
        return str(value)
