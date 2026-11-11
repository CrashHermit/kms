"""Composition for source discovery."""

from kms.config.settings import Settings
from kms.inference.predictors import PredictorFactory
from kms.source.discovery.discovery import SourceDiscovery
from kms.source.discovery.search import TavilyWebClient
from kms.source.discovery.selection import (
    DiscoverySelectionModule,
    DiscoverySelectionSignature,
    discovery_examples,
)


def build_source_discovery(
    settings: Settings,
    *,
    web: TavilyWebClient,
    predictors: PredictorFactory,
) -> SourceDiscovery:
    """Compose selector inference with an invocation-owned web adapter."""
    inference = settings.source_discovery.inference
    predictor = predictors.create(
        DiscoverySelectionModule,
        inference,
        DiscoverySelectionSignature,
        examples=discovery_examples(),
    )
    return SourceDiscovery(
        web,
        DiscoverySelectionModule(predictor),
    )
