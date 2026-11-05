"""Explicit runtime examples for text-only research actions."""

import dspy

from kms.core.model.research.inference import (
    FinishResearchAction,
    ResearchContext,
    ResearchSearchResult,
    ResearchTask,
    ResearchToolObservation,
    SubmitSourcesAction,
    WebSearchAction,
)


def _context(
    *,
    search_results: list[ResearchSearchResult] | None = None,
    observations: list[ResearchToolObservation] | None = None,
    tools_remaining: int = 8,
) -> ResearchContext:
    return ResearchContext(
        task=ResearchTask(question='Explain eigenvectors.', context=''),
        search_results=search_results or [],
        observations=observations or [],
        tools_remaining=tools_remaining,
    )


def researcher_examples() -> list[dspy.Example]:
    result = ResearchSearchResult(
        position=0,
        title='Linear Algebra Notes',
        url='https://example.edu/linear-algebra',
        snippet='A nonzero vector v is an eigenvector when Av = λv.',
    )
    return [
        dspy.Example(
            context=_context(),
            action=WebSearchAction(kind='web_search', query='eigenvectors'),
        ).with_inputs('context'),
        dspy.Example(
            context=_context(
                search_results=[result],
                observations=[
                    ResearchToolObservation(
                        kind='web_search',
                        summary='One university result is relevant.',
                    )
                ],
            ),
            action=SubmitSourcesAction(
                kind='submit_sources', result_positions=[0]
            ),
        ).with_inputs('context'),
        dspy.Example(
            context=_context(
                search_results=[result],
                observations=[
                    ResearchToolObservation(
                        kind='submit_sources',
                        summary='Persisted the selected university source.',
                        result_positions=[0],
                    )
                ],
                tools_remaining=1,
            ),
            action=FinishResearchAction(kind='finish'),
        ).with_inputs('context'),
        dspy.Example(
            context=_context(
                observations=[
                    ResearchToolObservation(
                        kind='invalid_action',
                        summary='Result position 3 was not offered.',
                    )
                ],
                tools_remaining=1,
            ),
            action=FinishResearchAction(kind='finish'),
        ).with_inputs('context'),
    ]
