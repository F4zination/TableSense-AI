from abc import ABC
from evaluator.metrics.exact_match_metric import ExactMatchMetric
from evaluator.metrics.bertscore_metric import BERTScoreMetric
from evaluator.metrics.rogue_metric import RogueMetric
from evaluator.metrics.metric import Metric


class Dataset(ABC):
    def __init__(self, dataset_path: str, is_remote: bool, metric: list[Metric]):
        self.dataset_path = dataset_path
        self.is_remote = is_remote
        self.metrics = metric


class SimpleTest(Dataset):
    def __init__(self):
        super().__init__(dataset_path="tab_llm_datasets/simple_test/dataset.py", is_remote=False,
                         metric=[ExactMatchMetric(), BERTScoreMetric(), RogueMetric()])


class WikiTableQuestions(Dataset):
    def __init__(self):
        super().__init__(dataset_path="TableSenseAI/WikiTableQuestions", is_remote=True, metric=[ExactMatchMetric(), BERTScoreMetric(),RogueMetric()])

class TabMWP(Dataset):
    def __init__(self):
        super().__init__(dataset_path="tab_llm_datasets/tabmwp/dataset.py", is_remote=False, metric=[ExactMatchMetric(), BERTScoreMetric(),RogueMetric()])
