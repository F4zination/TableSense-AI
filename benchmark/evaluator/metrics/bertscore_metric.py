from evaluate import load

from benchmark.evaluator.metrics.metric import Metric


class BERTScoreMetric(Metric):

    def __init__(self):
        super().__init__("BERTScore")

    def compute(self, predictions: list, references: list) -> float:
        exact_match = load("bertscore")
        return exact_match.compute(predictions=predictions, references=references, lang = "en")