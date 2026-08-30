"""LLM module base: the encode/decode boundary shared by every stage."""

import asyncio
import logging
import math
import numbers
import time

import dspy

from kms.core import llm, logs, recording, serve

logger = logging.getLogger(__name__)


def as_list(value: object) -> list:
    """Normalizes None, a bare item, or an iterable to a list.

    DSPy output fields come back as ``None``, a single item, or a list
    depending on the model, so every ``decode`` consumes list fields
    through this one helper instead of repeating ``list(x or [])``.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def require_bool(value: object, field_name: str) -> bool:
    """Returns an actual boolean prediction or raises a contract error."""
    if type(value) is not bool:
        raise ValueError(
            f'{field_name} must be a boolean, got {type(value).__name__}'
        )
    return value


def require_text(value: object, field_name: str) -> str:
    """Returns a non-empty text prediction or raises a contract error."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f'{field_name} must be a non-empty string, '
            f'got {type(value).__name__}'
        )
    return value


def require_number(
    value: object,
    field_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Validates a numeric prediction and optional inclusive bounds."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(
            f'{field_name} must be a number, got {type(value).__name__}'
        )
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{field_name} must be finite, got {value}')
    if minimum is not None and result < minimum:
        raise ValueError(
            f'{field_name} must be at least {minimum}, got {value}'
        )
    if maximum is not None and result > maximum:
        raise ValueError(f'{field_name} must be at most {maximum}, got {value}')
    return result


def require_positions(
    value: object,
    *,
    field_name: str,
    upper_bound: int,
    unique: bool = True,
    ordered: bool = False,
) -> list[int]:
    """Validates zero-based local positions without changing their meaning.

    ``upper_bound`` is exclusive. Callers should pass ``as_list`` output so
    DSPy cardinality normalization remains separate from contract validation.
    """
    if not isinstance(value, list):
        raise TypeError(
            f'{field_name} must be a list, got {type(value).__name__}'
        )
    for index, position in enumerate(value):
        if type(position) is not int:
            raise TypeError(
                f'{field_name}[{index}] must be an int, '
                f'got {type(position).__name__}'
            )
        if not 0 <= position < upper_bound:
            raise ValueError(
                f'{field_name}[{index}]={position} is outside '
                f'0..{upper_bound - 1}'
            )
    if unique and len(value) != len(set(value)):
        raise ValueError(f'{field_name} contains duplicate positions: {value}')
    if ordered and value != sorted(value):
        raise ValueError(f'{field_name} must be in document order: {value}')
    return value


class Module(dspy.Module):
    """Base class for a one-signature LLM module.

    A subclass declares :attr:`signature` and implements :meth:`encode`
    and :meth:`decode`, giving every module the same boundary: canonical
    inputs -> ``encode`` -> signature fields -> the LM -> ``decode`` ->
    canonical output. The base wires the shared plumbing — predictor
    construction, LM assignment, recording, and the sync ``forward``.
    """

    signature: type[dspy.Signature]
    record_name: str = 'module'

    def __init__(
        self,
        language_model: dspy.LM,
        recorder: recording.Recorder | None = None,
    ) -> None:
        super().__init__()
        self.predictor = dspy.Predict(self.signature)
        self.set_lm(language_model)
        self._recorder = recorder
        self._language_model = language_model

    def encode(self, **inputs: object) -> dict[str, object]:
        """Maps canonical inputs to signature-field kwargs."""
        raise NotImplementedError

    def decode(self, prediction: dspy.Prediction, **inputs: object) -> object:
        """Maps a prediction back to a canonical output."""
        raise NotImplementedError

    async def aforward(self, **inputs: object) -> object:
        """Encodes, ensures the configured model, calls, and decodes."""
        kwargs = self.encode(**inputs)
        start = time.perf_counter()
        prediction = await self._call_predictor(kwargs)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        output = self.decode(prediction, **inputs)
        if self._recorder:
            self._recorder.record(
                self.record_name,
                self.signature,
                kwargs,
                prediction,
                model=self._model_name(),
                duration_ms=duration_ms,
            )
        logger.debug(
            '%s: %d input(s) -> %s',
            self.record_name,
            len(inputs),
            logs.elide(str(output)),
        )
        return output

    async def _call_predictor(self, kwargs: dict[str, object]):
        """Calls the predictor under the configured model lease."""
        manager = serve.current_model_manager()
        if manager is None:
            return await self.predictor.acall(**kwargs)

        module_name = getattr(self._language_model, '_kms_module_name', None)
        if module_name is None:
            return await self.predictor.acall(**kwargs)
        model_id = llm.configured_serving_model(module_name)
        return await manager.aexecute(
            model_id,
            lambda: self.predictor.acall(**kwargs),
        )

    def _model_name(self) -> str | None:
        """Returns the module LM's model name, when known."""
        return getattr(self._language_model, 'model', None)

    def forward(self, **inputs: object) -> object:
        """Synchronous wrapper around :meth:`aforward`."""
        return asyncio.run(self.aforward(**inputs))
