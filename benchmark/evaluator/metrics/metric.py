from abc import ABC, abstractmethod


class Metric(ABC):
    @abstractmethod
    def compute(self, prediction: list, ground_truth: list) -> float:
        ...
