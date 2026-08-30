import pathlib
import sys
import types

import pytest

from kms import config

SRC = pathlib.Path(__file__).resolve().parent.parent / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def _uncached_settings(monkeypatch):
    """Makes settings re-read the environment on every access.

    Tests mutate env vars (via monkeypatch.setenv/delenv) and expect
    ``config.get_settings()`` consumers to reflect the change. The
    production ``get_settings`` is lru-cached, so swap it for the
    uncached builder during tests.
    """
    monkeypatch.setattr(config, 'get_settings', config.load_settings)


def _install_if_missing(name: str, build) -> None:
    try:
        __import__(name)
    except ImportError:
        for mod_name, mod in build().items():
            sys.modules[mod_name] = mod


def _dspy():
    module = types.ModuleType('dspy')

    class Image:
        def __init__(self, *a, **k):
            pass

    class Signature:
        pass

    class Module:
        def __init__(self, *a, **k):
            pass

        def set_lm(self, *a, **k):
            pass

    class Prediction:
        def __init__(self, **k):
            self.__dict__.update(k)

    class Example:
        def __init__(self, **k):
            self.__dict__.update(k)

        def with_inputs(self, *keys):
            self.__dict__['_input_keys'] = set(keys)
            return self

    class LM:
        def __init__(self, *a, **k):
            pass

    module.Image = Image
    module.Signature = Signature
    module.Module = Module
    module.Prediction = Prediction
    module.Example = Example
    module.LM = LM
    module.InputField = lambda *a, **k: None
    module.OutputField = lambda *a, **k: None
    module.Predict = lambda *a, **k: None
    return {'dspy': module}


def _pydantic():
    module = types.ModuleType('pydantic')

    class BaseModel:
        def __init__(self, **k):
            self.__dict__.update(k)

    module.BaseModel = BaseModel
    module.Field = lambda default=None, **k: default
    return {'pydantic': module}


def _langgraph_types():
    pkg = types.ModuleType('langgraph')
    pkg.__path__ = []
    sub = types.ModuleType('langgraph.types')

    class Send:
        def __init__(self, node, arg):
            self.node = node
            self.arg = arg

    sub.Send = Send
    return {'langgraph': pkg, 'langgraph.types': sub}


def _neo4j():
    module = types.ModuleType('neo4j')

    class AsyncDriver:
        async def close(self):
            pass

        async def verify_connectivity(self):
            pass

        def session(self, **k):
            raise RuntimeError('stub neo4j driver has no live session')

    class AsyncGraphDatabase:
        @staticmethod
        def driver(*a, **k):
            return AsyncDriver()

    module.AsyncDriver = AsyncDriver
    module.AsyncGraphDatabase = AsyncGraphDatabase
    return {'neo4j': module}


_install_if_missing('dspy', _dspy)
_install_if_missing('pydantic', _pydantic)
_install_if_missing('langgraph.types', _langgraph_types)
_install_if_missing('neo4j', _neo4j)
