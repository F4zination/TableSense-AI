from abc import ABC, abstractmethod


class Metric(ABC):

    def __init__(self, metric_name):
        self.metric_name = metric_name

    @abstractmethod
    def compute(self, predictions: list, references: list) -> float:
        ...
