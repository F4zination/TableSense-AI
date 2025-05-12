from evaluate import load

from benchmark.evaluator.metrics.metric import Metric


class ExactMatchMetric(Metric):
    def compute(self, prediction: list, ground_truth: list) -> float:
        exact_match = load("exact_match")
        return exact_match.compute(predictions=[prediction], references=[ground_truth])
