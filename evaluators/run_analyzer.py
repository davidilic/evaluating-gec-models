import json, os
from typing import List, Dict
from metrics.metric import EvaluationMetric, EvalResult


class RunAnalyzer:
    """
    Analyzes model outputs against target data points using specified evaluation metrics.
    """

    def __init__(self, metrics: List[EvaluationMetric], raw_output_path: str):
        self.metrics = metrics
        self.raw_output_path = raw_output_path
        self.results = {}

    def analyze(self) -> Dict[str, Dict[str, float]]:
        """
        Analyzes the loaded evaluation results using all specified metrics for each model.
        """
        eval_results_by_model = self.load_eval_results()

        for model_name, eval_results in eval_results_by_model.items():
            model_results = {}
            for metric in self.metrics:
                metric_name = metric.__class__.__name__
                score = metric.calculate(eval_results)
                model_results[metric_name] = score
            self.results[model_name] = model_results

        return self.results

    def load_eval_results(self) -> Dict[str, List[EvalResult]]:
        """
        Loads and parses the raw output file, converting it into a dictionary of EvalResult lists
        grouped by model.
        """
        eval_results_by_model = {}
        with open(self.raw_output_path, "r", encoding="utf-8") as file:
            data_points = json.load(file)

        for entry in data_points:
            data_point = entry["data_point"]
            source_text = data_point["source"]
            reference_text = data_point["target"]

            for model_name, model_output in entry["model_output"].items():
                eval_result = EvalResult(
                    original_text=source_text, model_output=model_output, reference_correction=reference_text
                )

                if model_name not in eval_results_by_model:
                    eval_results_by_model[model_name] = []
                eval_results_by_model[model_name].append(eval_result)

        return eval_results_by_model

    def save_results(self, output_path: str) -> None:
        """
        Saves the calculated metrics to a specified path in JSON format.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(self.results, file, indent=4)
