from metrics.metric import EvalResult, EvaluationMetric
from typing import List
from bert_score import score
import torch


class BERTScoreMetric(EvaluationMetric):

    def __init__(self, scorer: str = "distilbert-base-uncased", batch_size: int = 32, lang: str = "en"):
        self.scorer = scorer
        self.batch_size = batch_size
        self.lang = lang
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def calculate(self, eval_results: List[EvalResult]) -> float:
        """Calculate corpus-level BERTScore"""

        model_outputs = [result.model_output for result in eval_results]
        references = [result.reference_correction for result in eval_results]

        P, R, F1 = score(
            model_outputs,
            references,
            model_type=self.scorer,
            batch_size=self.batch_size,
            device=self.device,
            lang=self.lang,
        )

        return F1.mean().item()

    def calculate_single_example(self, eval_result: EvalResult) -> float:
        """Calculate BERTScore for single example"""

        P, R, F1 = score(
            [eval_result.model_output],
            [eval_result.reference_correction],
            model_type=self.model_type,
            batch_size=1,
            device=self.device,
            lang=self.lang,
        )

        return F1.item()
