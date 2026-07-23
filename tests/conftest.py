"""Test setup: make `kms.*` importable and stub the heavy third-party deps.

The unit suite exercises pure pipeline logic (flattening, windowing, the finder
cursor-walk, entity output) and must run without installing dspy/pydantic/langgraph.
Each stub is installed ONLY if the real package is missing, so the same tests also run
unchanged in a full environment where the real deps are present.
"""

import pathlib
import sys
import types

# `pyproject` sets package=false and the code imports itself as `kms.*`, so the
# package root is `src/`. Put it on the path for the tests.
SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _install_if_missing(name: str, build) -> None:
    try:
        __import__(name)
    except ImportError:
        for mod_name, mod in build().items():
            sys.modules[mod_name] = mod


def _dspy():
    m = types.ModuleType("dspy")

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
            self.__dict__["_input_keys"] = set(keys)
            return self

    class LM:
        def __init__(self, *a, **k):
            pass

    m.Image = Image
    m.Signature = Signature
    m.Module = Module
    m.Prediction = Prediction
    m.Example = Example
    m.LM = LM
    m.InputField = lambda *a, **k: None
    m.OutputField = lambda *a, **k: None
    m.ChainOfThought = lambda *a, **k: None
    m.Predict = lambda *a, **k: None
    return {"dspy": m}


def _pydantic():
    m = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, **k):
            self.__dict__.update(k)

    m.BaseModel = BaseModel
    m.Field = lambda default=None, **k: default
    return {"pydantic": m}


def _langgraph_types():
    pkg = types.ModuleType("langgraph")
    pkg.__path__ = []  # mark as a package so `langgraph.types` can attach
    sub = types.ModuleType("langgraph.types")

    class Send:
        def __init__(self, node, arg):
            self.node = node
            self.arg = arg

    sub.Send = Send
    return {"langgraph": pkg, "langgraph.types": sub}


def _neo4j():
    # Enough surface for `kms.graph.db` / `kms.graph.schema` to import and for the pure
    # helpers (schema_statements, vector_dim) to be unit-tested with no server. Anything
    # that actually talks to a database is exercised only by the opt-in integration test,
    # which skips unless NEO4J_URI is set.
    m = types.ModuleType("neo4j")

    class AsyncDriver:
        async def close(self):
            pass

        async def verify_connectivity(self):
            pass

        def session(self, **k):
            raise RuntimeError("stub neo4j driver has no live session")

    class AsyncGraphDatabase:
        @staticmethod
        def driver(*a, **k):
            return AsyncDriver()

    m.AsyncDriver = AsyncDriver
    m.AsyncGraphDatabase = AsyncGraphDatabase
    return {"neo4j": m}


_install_if_missing("dspy", _dspy)
_install_if_missing("pydantic", _pydantic)
_install_if_missing("langgraph.types", _langgraph_types)
_install_if_missing("neo4j", _neo4j)
