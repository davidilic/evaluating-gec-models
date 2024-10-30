from metrics.errant import ERRANTMetricF05, ERRANTMetricRecall, ERRANTMetricPrecision
from models.groq_model import Gemma9bGEC, Llama90bGEC, Llama11bGEC, Mixtral8x7bGEC
from evaluators.run_analyzer import RunAnalyzer
from models.language_tool import LanguageTool
from evaluators.eval_runner import EvalRunner
from metrics.bertscore import BERTScoreMetric
from data.data_utils import load_gec_data
from metrics.gleu import GLEUMetric

# Datasets
bea2019_dataset = load_gec_data("data/bea2019/bea2019_processed.jsonl")
fm_dataset = load_gec_data("data/falko_merlin/fm_processed.jsonl")

# Models
models = [LanguageTool(), Gemma9bGEC(), Llama90bGEC(), Llama11bGEC(), Mixtral8x7bGEC()]

# Metrics
errant_precision = ERRANTMetricPrecision()
errant_recall = ERRANTMetricRecall()
errant_f05 = ERRANTMetricF05()
gleu = GLEUMetric()
bertscore_english = BERTScoreMetric()
bertscore_german = BERTScoreMetric(lang="de")

# Evaluation
bea2019_output_path = "results/raw/bea2019_raw.jsonl"
bea2019_runner = EvalRunner(bea2019_dataset, models)
bea2019_runner.run_evaluation()
bea2019_runner.save_results(bea2019_output_path)

bea2019_metrics = [errant_f05, errant_recall, errant_precision, gleu, bertscore_english]
bea2019_analyzer = RunAnalyzer(bea2019_metrics, bea2019_output_path)
bea2019_analyzer.analyze()
bea2019_analyzer.save_results("results/analyzed/bea2019_analyzed.jsonl")

fm_output_path = "results/raw/fm_raw.jsonl"
fm_runner = EvalRunner(fm_dataset, models)
fm_runner.run_evaluation()
fm_runner.save_results(fm_output_path)

fm_metrics = [gleu, bertscore_german]
fm_analyzer = RunAnalyzer(fm_metrics, fm_output_path)
fm_analyzer.analyze()
fm_analyzer.save_results("results/analyzed/fm_analyzed.jsonl")
