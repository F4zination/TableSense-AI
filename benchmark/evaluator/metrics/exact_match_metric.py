from evaluate import load

from benchmark.evaluator.metrics.metric import Metric


class ExactMatchMetric(Metric):

    def __init__(self):
        super().__init__("Exact match")

    def compute(self, predictions: list, references: list) -> float:
        exact_match = load("exact_match")
        return exact_match.compute(predictions=predictions, references=references)
