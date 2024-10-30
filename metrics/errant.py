from metrics.metric import EvaluationMetric, EvalResult
import spacy, os, tempfile, subprocess
from typing import Tuple, List


class ERRANTMetric(EvaluationMetric):

    def __init__(self, type: str = "f0.5"):
        self.nlp = spacy.load("en_core_web_sm")
        self.type = type

    def calculate(self, eval_results: List[EvalResult]) -> float:
        """Calculate aggregated ERRANT F0.5 score."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = self._write_texts_to_temp_files(eval_results, tmp_dir)
            reference_m2_path = os.path.join(tmp_dir, "reference.m2")
            self._execute_errant_command("parallel", orig=paths[0], cor=paths[1], out=reference_m2_path)
            return self._compute_errant_score(reference_m2_path, paths[2], tmp_dir)

    def _write_texts_to_temp_files(self, eval_results: List[EvalResult], tmp_dir: str) -> Tuple[str, str, str]:
        """Write original, reference, and model texts to temporary files."""
        filenames = ["original_text.txt", "reference_correction.txt", "model_output.txt"]
        paths = [os.path.join(tmp_dir, name) for name in filenames]

        with open(paths[0], "w") as orig, open(paths[1], "w") as ref, open(paths[2], "w") as model:
            for result in eval_results:
                files = [orig, ref, model]
                texts = [result.original_text, result.reference_correction, result.model_output]
                for file, text in zip(files, texts):
                    file.write(text.replace("\n", " ") + "\n")

        return tuple(paths)

    def _compute_errant_score(self, m2_path: str, model_output_path: str, tmp_dir: str) -> float:
        """Calculate precision, recall or F0.5 score using ERRANT."""
        result = self._run_errant_evaluation(m2_path, model_output_path, tmp_dir)

        if self.type == "precision":
            return float(result.stdout.split("\n")[3].split()[-3])
        elif self.type == "recall":
            return float(result.stdout.split("\n")[3].split()[-2])
        else:
            return float(result.stdout.split("\n")[3].split()[-1])

    def _run_errant_evaluation(self, m2_path: str, model_output_path: str, tmp_dir: str):
        """Process files and run ERRANT evaluation."""
        original_text_path = self._extract_original_from_m2(m2_path, tmp_dir)
        processed_paths = self._prepare_model_output(model_output_path, tmp_dir)

        self._execute_errant_command(
            "parallel", orig=original_text_path, cor=processed_paths[0], out=processed_paths[1]
        )
        return self._execute_errant_command("compare", hyp=processed_paths[1], ref=m2_path)

    def _extract_original_from_m2(self, m2_path: str, tmp_dir: str) -> str:
        """Extract original texts from M2 format file."""
        output_path = os.path.join(tmp_dir, "extracted_original.txt")
        with open(m2_path) as f, open(output_path, "w") as out:
            sentences = [line[2:].strip() for line in f if line.startswith("S ")]
            out.write("\n".join(sentences) + "\n")
        return output_path

    def _prepare_model_output(self, model_output_path: str, tmp_dir: str) -> Tuple[str, str]:
        """Process model output for ERRANT evaluation."""
        processed_path = os.path.join(tmp_dir, "processed_output.txt")
        m2_output_path = os.path.join(tmp_dir, "model_m2.m2")

        with open(model_output_path) as f, open(processed_path, "w") as out:
            out.write(f.read())
        return processed_path, m2_output_path

    def _execute_errant_command(self, command: str, **kwargs) -> subprocess.CompletedProcess:
        """Execute an ERRANT CLI command."""
        args = " ".join(f'-{k} "{v}"' for k, v in kwargs.items())
        return subprocess.run(f"errant_{command} {args}", shell=True, check=True, capture_output=True, text=True)


class ERRANTMetricF05(ERRANTMetric):
    def __init__(self):
        super().__init__(type="f0.5")


class ERRANTMetricRecall(ERRANTMetric):
    def __init__(self):
        super().__init__(type="recall")


class ERRANTMetricPrecision(ERRANTMetric):
    def __init__(self):
        super().__init__(type="precision")
