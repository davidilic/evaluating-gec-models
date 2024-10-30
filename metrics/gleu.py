import nltk
from typing import List
from nltk.tokenize import word_tokenize
from metrics.metric import EvalResult, EvaluationMetric


class GLEUMetric(EvaluationMetric):

    def __init__(self, min_len: int = 1, max_len: int = 4):
        self.min_len = min_len
        self.max_len = max_len

    def calculate(self, eval_results: List[EvalResult]) -> float:
        """Calculate corpus-level GLEU score"""
        hypotheses, references = self._tokenize_texts(eval_results)

        score = nltk.translate.gleu_score.corpus_gleu(
            references, hypotheses, min_len=self.min_len, max_len=self.max_len
        )

        return score

    def calculate_single_example(self, eval_result: EvalResult) -> float:
        """Calculate GLEU score for single example"""
        hyp_tokens = word_tokenize(eval_result.model_output.lower())
        ref_tokens = word_tokenize(eval_result.reference_correction.lower())

        score = nltk.translate.gleu_score.sentence_gleu(
            [ref_tokens], hyp_tokens, min_len=self.min_len, max_len=self.max_len
        )

        return score

    def _tokenize_texts(self, eval_results: List[EvalResult]):
        """Tokenize texts for GLEU calculation"""
        hypotheses = []
        references = []

        for result in eval_results:
            hyp_tokens = word_tokenize(result.model_output.lower())
            hypotheses.append(hyp_tokens)

            ref_tokens = word_tokenize(result.reference_correction.lower())
            references.append([ref_tokens])

        return hypotheses, references
