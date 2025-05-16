from typing import List

from evaluate import load, EvaluationModule

from benchmark.evaluator.dataset_definition import Dataset
from benchmark.evaluator.metrics.metric import Metric


class EvalConfig:
    """
    Configuration for the evaluation process.

    Args:
        datasets (List[Dataset]): A list of dataset enums to evaluate on.
        metrics (List[Metric]): A list of metrics from the Metric class implemented.
        force_redownload (bool, optional): Whether to force redownloading datasets. Defaults to False.
            Usefully if you want to get sure, that the newest version of the dataset will be used.
        verbose (bool, optional): if true for every question the prediction of the model and the ground truth will be printed.

    Attributes:
        datasets (List[Dataset]): Datasets to evaluate.
        metrics (List[Metric]): A list of metric functions for evaluation.
        force_redownload (bool): Flag to force dataset download.
        verbose (bool): Flag to force dataset download.
    """

    def __init__(self, datasets: List[Dataset], metrics: List[Metric], force_redownload: bool = False, verbose: bool = False):
        self.datasets = datasets
        self.force_redownload = force_redownload
        self.metrics = metrics or []
        self.verbose = verbose
