from evaluate import load

from benchmark.evaluator.metrics.metric import Metric


class RogueMetric(Metric):

    def __init__(self):
        super().__init__("Rogue Score")

    def compute(self, predictions: list, references: list) -> float:
        exact_match = load("rouge")
        return exact_match.compute(predictions=predictions, references=references)
