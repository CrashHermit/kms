"""Pure composition of source-local construction records."""

from dataclasses import dataclass

from kms.core import models


@dataclass(frozen=True, slots=True)
class ComposedPart:
    """One ordered source node in a composed statement."""

    position: int
    type: models.NodeType | None
    content: str | None
    document_index: int | None
    image_path: str | None


@dataclass(frozen=True, slots=True)
class ComposedContent:
    """Ordered, source-local composition of source-node members."""

    source: str
    parts: tuple[ComposedPart, ...]

    @property
    def text(self) -> str:
        """Returns non-image member text using the legacy separator."""
        return '\n\n'.join(
            part.content
            for part in self.parts
            if part.type != models.NodeType.IMAGE and part.content
        )

    @property
    def pictures(self) -> list[dict[str, int | str | None]]:
        """Returns image metadata in composed document order."""
        return [
            {
                'index': index,
                'document_index': part.document_index,
                'image_path': part.image_path,
            }
            for index, part in enumerate(
                part
                for part in self.parts
                if part.type == models.NodeType.IMAGE
            )
        ]


@dataclass(frozen=True, slots=True)
class ComposedStatement(ComposedContent):
    """Ordered composition of one statement's source members."""


@dataclass(frozen=True, slots=True)
class ComposedProcedure:
    """Ordered composition and context for one procedure."""

    procedure: models.Procedure
    content: ComposedContent
    statement: ComposedStatement | None
    statement_uuid: str | None


def _compose_content(
    bundle: models.ConstructionBundle,
    members: list[int],
    label: str,
    content_type: type[ComposedContent] = ComposedContent,
) -> ComposedContent:
    """Resolves and orders source members into typed content parts."""
    if len(members) != len(set(members)):
        raise ValueError(f'{label} contains duplicate member positions')
    missing = [pos for pos in members if pos >= len(bundle.nodes) or pos < 0]
    if missing:
        raise ValueError(
            f'{label} references missing node positions: {missing!r}'
        )
    parts = tuple(
        ComposedPart(
            position=position,
            type=bundle.nodes[position].type,
            content=bundle.nodes[position].content,
            document_index=bundle.nodes[position].document_index,
            image_path=bundle.nodes[position].image_path,
        )
        for position in sorted(members)
    )
    return content_type(source=bundle.source.key or '', parts=parts)


def compose_statement(
    bundle: models.ConstructionBundle,
    statement: models.Statement,
) -> ComposedStatement:
    """Composes a statement from its explicit source-node members.

    Member IDs are resolved against the bundle and ordered by their stable
    flattened node IDs. The source nodes and statement are never mutated.
    """
    return _compose_content(
        bundle,
        statement.member_positions,
        'statement',
        ComposedStatement,
    )


def compose_procedure(
    bundle: models.ConstructionBundle,
    procedure: models.Procedure,
) -> ComposedProcedure:
    """Composes one procedure without persistence, lookup, or model calls."""
    content = _compose_content(bundle, procedure.member_positions, 'procedure')

    matching_statements = [
        statement
        for statement in bundle.statements
        if statement.block == procedure.block
    ]
    if procedure.statement_uuid is not None:
        if not procedure.statement_uuid:
            raise ValueError('procedure statement_uuid must be non-empty')
        statement = next(
            (
                statement
                for statement in bundle.statements
                if statement.uuid == procedure.statement_uuid
            ),
            None,
        )
        if statement is None:
            raise ValueError(
                f'procedure references unknown statement UUID '
                f'{procedure.statement_uuid!r}'
            )
        composed_statement = compose_statement(bundle, statement)
    elif len(matching_statements) > 1:
        raise ValueError('procedure has ambiguous statement linkage')
    elif matching_statements:
        composed_statement = compose_statement(bundle, matching_statements[0])
    else:
        composed_statement = None

    return ComposedProcedure(
        procedure=procedure,
        content=content,
        statement=composed_statement,
        statement_uuid=procedure.statement_uuid,
    )
