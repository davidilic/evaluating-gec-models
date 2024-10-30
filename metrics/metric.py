from typing import Protocol
from dataclasses import dataclass


@dataclass
class EvalResult:
    original_text: str
    model_output: str
    reference_correction: str


class EvaluationMetric(Protocol):

    def calculate(self, eval_results: list[EvalResult]) -> float:
        """Calculate metric score"""
        ...
