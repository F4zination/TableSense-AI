from evaluate import load

from benchmark.evaluator.metrics.metric import Metric


class BERTScoreMetric(Metric):

    def __init__(self):
        super().__init__("BERTScore")

    def compute(self, predictions: list, references: list) -> float:
        bertScore = load("bertscore")
        return bertScore.compute(predictions=predictions, references=references, lang = "en")