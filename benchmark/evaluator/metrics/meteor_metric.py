from evaluate import load

from benchmark.evaluator.metrics.metric import Metric


class MeteorMetric(Metric):

    def __init__(self):
        super().__init__("METEOR Score")

    def compute(self, predictions: list, references: list) -> float:
        exact_match = load("meteor")
        return exact_match.compute(predictions=predictions, references=references)
