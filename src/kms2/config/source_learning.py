"""Configuration for typed source-learning inference phases."""

from pydantic import BaseModel, Field

from kms2.config.inference import TextInferenceSettings


class SourceLearningSettings(BaseModel):
    """Independent model settings for each source-learning boundary."""

    source_entity_learning_fact: TextInferenceSettings = Field(
        default_factory=TextInferenceSettings
    )
    source_entity_flashcard: TextInferenceSettings = Field(
        default_factory=TextInferenceSettings
    )
    source_event_learning_fact: TextInferenceSettings = Field(
        default_factory=TextInferenceSettings
    )
    source_event_flashcard: TextInferenceSettings = Field(
        default_factory=TextInferenceSettings
    )
    source_predicate_learning_fact: TextInferenceSettings = Field(
        default_factory=TextInferenceSettings
    )
    source_predicate_flashcard: TextInferenceSettings = Field(
        default_factory=TextInferenceSettings
    )
    source_triplet_learning_fact: TextInferenceSettings = Field(
        default_factory=TextInferenceSettings
    )
    source_triplet_flashcard: TextInferenceSettings = Field(
        default_factory=TextInferenceSettings
    )
