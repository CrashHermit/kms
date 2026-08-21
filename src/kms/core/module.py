"""LLM module base: the encode/decode boundary shared by every stage."""

import asyncio
import logging
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
    use_chain_of_thought: bool = False

    def __init__(
        self,
        language_model: dspy.LM,
        recorder: recording.Recorder | None = None,
    ) -> None:
        super().__init__()
        predictor_factory = (
            dspy.ChainOfThought if self.use_chain_of_thought else dspy.Predict
        )
        self.predictor = predictor_factory(self.signature)
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
        if self._recorder:
            self._recorder.record(
                self.record_name,
                self.signature,
                kwargs,
                prediction,
                model=self._model_name(),
                duration_ms=duration_ms,
            )
        output = self.decode(prediction, **inputs)
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
