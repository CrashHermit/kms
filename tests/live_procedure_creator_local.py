import asyncio

from kms.construction import procedure_creator
from kms.core import content, llm, serve

STATEMENT = (
    'Prove that every induced subgraph of a graph $G$ is also a subgraph '
    'of $G$.'
)
ENTITY_DEFS = (
    "subgraph: $G' = (V', E')$ is a subgraph of $G = (V, E)$ if "
    "$V' \\subseteq V$ and $E' \\subseteq E$.\n"
    "induced subgraph: $G' = (V', E')$ is an induced subgraph of $G$ if "
    "$V' \\subseteq V$ and $E'$ contains every edge of $G$ whose endpoints "
    "are in $V'$."
)


async def main():
    manager = serve.RouterManager(serve.default_router())
    try:
        manager.switch('procedure_creator')
        language_model = llm.module_lm('procedure_creator')
        writer = procedure_creator.ProcedureWriter(language_model)
        parts = content.Content.from_text(STATEMENT)
        procedure = await writer.aforward(
            parts=parts, entity_definitions=ENTITY_DEFS
        )
        print('PROCEDURE:\n', procedure)
        splitter = procedure_creator.StepSplitter(language_model)
        steps = await splitter.aforward(procedure=procedure)
        for index, step in enumerate(steps, 1):
            print(f'{index}. {step}')
    finally:
        manager.shutdown()


if __name__ == '__main__':
    asyncio.run(main())
