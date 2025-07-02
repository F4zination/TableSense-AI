from evaluate import load
import numpy as np

from benchmark.evaluator.metrics.metric import Metric


class BERTScoreMetric(Metric):

    def __init__(self):
        super().__init__("BERTScore")

    def compute(self, predictions: list, references: list) -> float:
        bertScore = load("bertscore")
        result = bertScore.compute(predictions=predictions, references=references, lang = "en")
        result["precision"] = float(np.mean(result["precision"]))
        result["f1"] = float(np.mean(result["f1"]))
        result["recall"] = float(np.mean(result["recall"]))
        return result