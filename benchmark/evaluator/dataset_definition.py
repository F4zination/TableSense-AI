from abc import ABC

from benchmark.evaluator.metrics.exact_match_metric import ExactMatchMetric
from benchmark.evaluator.metrics.bertscore_metric import BERTScoreMetric
from benchmark.evaluator.metrics.meteor_metric import MeteorMetric
from benchmark.evaluator.metrics.rogue_metric import RogueMetric
from benchmark.evaluator.metrics.metric import Metric


class Dataset(ABC):
    """
    Enum for available datasets.

    Attributes:
        SimpleTest (str): Path to the simple test dataset script.
    """

    def __init__(self, dataset_path: str, is_remote: bool, metric: list[Metric]):
        self.dataset_path = dataset_path
        self.is_remote = is_remote
        self.metrics = metric


class SimpleTest(Dataset):
    def __init__(self):
        super().__init__(dataset_path="tab_llm_datasets/simple_test/dataset.py", is_remote=False,
                         metric=[ExactMatchMetric(), BERTScoreMetric(), MeteorMetric(), RogueMetric()])


class WikiTableQuestions(Dataset):
    def __init__(self):
        super().__init__(dataset_path="TableSenseAI/WikiTableQuestions", is_remote=True,
                         metric=[ExactMatchMetric(), BERTScoreMetric(), RogueMetric()])


class FreeformTableQA(Dataset):
    def __init__(self):
        super().__init__(dataset_path="TableSenseAI/FreeformTableQA", is_remote=True,
                         metric=[ExactMatchMetric(), BERTScoreMetric(), RogueMetric()])
