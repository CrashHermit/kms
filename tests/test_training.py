from types import SimpleNamespace

import dspy
import pytest

from kms.training import trainer as training


def test_comparator_produces_exact_match_and_percentage_similarity():
    comparator = training.GenericComparator()
    exact = comparator.compare({'b': 2, 'a': 1}, {'a': 1, 'b': 2})
    assert exact.exact_match is True
    assert exact.score == 100
    assert exact.diff == ''

    different = comparator.compare('expected text', 'actual text')
    assert different.exact_match is False
    assert 0 < different.score < 100
    assert 'expected' in different.diff
    assert 'predicted' in different.diff

    assert 0 < different.score < 100

def test_metric_uses_only_character_difference_by_default():
    evaluator = training.Evaluator()
    example = dspy.Example(text='hello', answer='HELLO').with_inputs('text')
    assert evaluator.metric(example, 'HELLO') == 100
    assert evaluator.metric(example, 'HELL') < 100


def test_metric_rejects_multiple_output_fields():
    evaluator = training.Evaluator()
    example = dspy.Example(text='hello', answer='HELLO', extra='x').with_inputs(
        'text'
    )
    with pytest.raises(ValueError, match='exactly one output field'):
        evaluator.metric(example, 'HELLO')


def test_optional_judge_can_be_enabled_explicitly():
    class FakeJudge:
        def judge(self, **kwargs):
            return training.JudgeResult(0.25, False, 'Needs work.')

    evaluator = training.Evaluator(
        judge=FakeJudge(), deterministic_weight=0.4
    )
    result = evaluator.metric(
        dspy.Example(text='input', answer='gold').with_inputs('text'),
        'gold',
    )
    assert result == pytest.approx(0.4 * 100 + 0.6 * 0.25)


def test_simple_judge_decodes_validated_prediction():
    judge = training.SimpleJudge.__new__(training.SimpleJudge)
    result = judge.decode(
        SimpleNamespace(score=0.75, passed=True, feedback='Good.')
    )
    assert result == training.JudgeResult(0.75, True, 'Good.')

    with pytest.raises(ValueError, match='score must be at most 1'):
        judge.decode(SimpleNamespace(score=2, passed=True, feedback='Bad.'))


def test_golden_data_remains_a_plain_dspy_dataset_artifact(tmp_path):
    path = tmp_path / 'facts.jsonl'
    path.write_text(
        '{"id":"one","source":"book.pdf","target_text":"Text.",'
        '"facts":["Text."]}\n'
    )
    records = [
        __import__('json').loads(line)
        for line in path.read_text().splitlines()
    ]
    assert records[0]['source'] == 'book.pdf'
    assert records[0]['facts'] == ['Text.']

def test_compile_uses_explicit_teacher_lm(monkeypatch):
    class Program(dspy.Module):
        def __init__(self, language_model):
            super().__init__()
            self.predictor = dspy.Predict('text -> answer')
            self.set_lm(language_model)

    student_lm = dspy.LM(
        'openai/student',
        api_base='http://student/v1',
        api_key='not-needed',
        max_tokens=17,
        cache=False,
    )
    teacher_lm = dspy.LM(
        'openai/teacher',
        api_base='http://teacher/v1',
        api_key='not-needed',
        max_tokens=23,
        cache=False,
    )
    student = Program(student_lm)
    teacher = Program(student_lm)
    captured = {}

    class Optimizer:
        def __init__(self, **kwargs):
            captured['optimizer'] = kwargs

        def compile(self, program, *, teacher=None, trainset):
            captured['program'] = program
            captured['teacher'] = teacher
            return program

    monkeypatch.setattr(dspy, 'BootstrapFewShot', Optimizer)
    result = training.Evaluator().compile(
        student,
        [dspy.Example(text='input', answer='output').with_inputs('text')],
        teacher=teacher,
        teacher_lm=teacher_lm,
    )

    assert result is student
    assert captured['teacher'] is teacher
    assert teacher.predictor.lm is teacher_lm
    assert student.predictor.lm is student_lm
    assert 'teacher_settings' not in captured['optimizer']
