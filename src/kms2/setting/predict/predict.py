from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class PredictorStrategy(StrEnum):
    PREDICT = 'predict'
    CHAIN_OF_THOUGHT = 'chain_of_thought'
    REACT = 'react'
    BEST_OF_N = 'best_of_n'
    CODE_ACT = 'code_act'
    MULTI_CHAIN_COMPARISON = 'multi_chain_comparison'
    PROGRAM_OF_THOUGHT = 'program_of_thought'


class Predictor(BaseModel):
    model_config = ConfigDict(extra='forbid')

    strategy: PredictorStrategy = Field(default=PredictorStrategy.PREDICT)
