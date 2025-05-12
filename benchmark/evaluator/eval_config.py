from typing import List

from evaluate import load, EvaluationModule

from benchmark.evaluator.dataset_definition import Dataset


class EvalConfig:
    """
    Configuration for the evaluation process.

    Args:
        datasets (List[Dataset]): A list of dataset enums to evaluate on.
        force_redownload (bool, optional): Whether to force redownloading datasets. Defaults to False.
            Usefully if you want to get sure, that the newest version of the dataset will be used.
        metrics (List[Callable], optional): A list of metric functions to evaluate the predictions. Defaults to None.

    Attributes:
        datasets (List[Dataset]): Datasets to evaluate.
        force_redownload (bool): Flag to force dataset download.
        metrics (List[Callable]): A list of metric functions for evaluation.
    """

    def __init__(self, datasets: List[Dataset], force_redownload: bool = False, metrics: List[EvaluationModule] = None):
        self.datasets = datasets
        self.force_redownload = force_redownload
        self.metrics = metrics or []

exact_match_metric = load("exact_match")

