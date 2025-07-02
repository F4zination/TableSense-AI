from evaluate import load
import numpy as np

from benchmark.evaluator.metrics.metric import Metric


class BERTScoreMetric(Metric):

    def __init__(self):
        super().__init__("BERTScore")

    def compute(self, predictions: list, references: list) -> dict:
        bertScore = load("bertscore")
        scores = bertScore.compute(predictions=predictions, references=references, lang = "en")

        averaged_scores = {
            "precision": sum(scores["precision"]) / len(scores["precision"]),
            "recall": sum(scores["recall"]) / len(scores["recall"]),
            "f1": sum(scores["f1"]) / len(scores["f1"]),
        }

        return averaged_scores