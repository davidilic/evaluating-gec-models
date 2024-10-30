import json, os
from models.gec_model import GECModel
from torch.utils.data import DataLoader


class EvalRunner:
    """
    EvalRunner generates error correction predictions for a dataset using a list of GEC models.
    It saves these predictions to a JSON file for further analysis.
    """

    def __init__(self, dataset: DataLoader, models: list[GECModel]):
        self.dataset = dataset
        self.models = models
        self.results = []

    def run_evaluation(self):
        """
        Run the evaluation on the dataset using the provided models.
        """
        for data_point in self.dataset:
            data_results = {"data_point": data_point, "model_output": {}}
            print(f"Processing data point: {data_point}")
            input_text, language = data_point["source"], data_point["language"]
            for model in self.models:
                prediction = model.correct_errors(input_text, language)
                model_name = model.__class__.__name__
                data_results["model_output"][model_name] = prediction
            self.results.append(data_results)

    def save_results(self, output_path: str):
        """
        Save the evaluation results to a JSON file.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as out_file:
            json.dump(self.results, out_file, ensure_ascii=False, indent=4)
