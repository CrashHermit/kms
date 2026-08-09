from typing import TYPE_CHECKING

__all__ = ['run']

if TYPE_CHECKING:
    from kms.pipeline import run


def __getattr__(name: str) -> object:
    if name == 'run':
        from kms.pipeline import run

        return run
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

