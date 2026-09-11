import asyncio

from kms2.database.schema import SCHEMA_STATEMENTS, ensure_schema


class _RecordingSession:
    def __init__(self) -> None:
        self.statements: list[str] = []

    async def run(self, statement: str) -> None:
        self.statements.append(statement)


class _SessionContext:
    def __init__(self, session: _RecordingSession) -> None:
        self._session = session

    async def __aenter__(self) -> _RecordingSession:
        return self._session

    async def __aexit__(self, *args: object) -> None:
        return None


def test_ensure_schema_runs_only_kms2_uuid_constraints():
    session = _RecordingSession()

    asyncio.run(ensure_schema(lambda: _SessionContext(session)))

    assert session.statements == list(SCHEMA_STATEMENTS)
    assert len(session.statements) == 3
    schema = '\n'.join(session.statements)
    assert 'SourcePage' not in schema
    assert 'SourceBlock' in schema
    assert 'VisualAsset' in schema
    assert 'HAS_PAGE' not in schema
    assert 'CONTAINS_BLOCK' not in schema
