from evaluate import load

from benchmark.evaluator.metrics.metric import Metric


class RogueMetric(Metric):

    def __init__(self):
        super().__init__("Rogue Score")

    def compute(self, predictions: list, references: list) -> float:
        rogue_score = load("rouge")
        return rogue_score.compute(predictions=predictions, references=references)
