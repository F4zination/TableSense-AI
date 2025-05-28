from pathlib import Path
from typing import List

from datasets import load_dataset
from tqdm import tqdm

from benchmark.evaluator.metrics.metric import Metric
from tablesense_ai.agent.base import BaseAgent
from benchmark.evaluator.eval_config import EvalConfig


class Evaluator:
    """
    Runs evaluation of an LLM agent on one or more datasets.

    This class loads the specified datasets and uses the given agent
    to generate predictions, comparing them to ground-truth answers.

    Args:
        config (EvalConfig): Evaluation configuration specifying datasets and other behavior.
        agent (Agent): The model or agent responsible for generating predictions.

    Attributes:
        predictor (Agent): The evaluation agent.
        datasets (List[DatasetDict]): Loaded datasets ready for evaluation.
    """

    def __init__(self, config: EvalConfig, agent: BaseAgent):
        self.verbose = config.verbose
        self.predictor = agent
        self.datasets = []

        if config.force_redownload is True:
            download_mode = "force_redownload"
        else:
            download_mode = "reuse_dataset_if_exists"

        current_file_path = Path(__file__).resolve().parent  # path to evaluator.py
        base_dir = current_file_path.parent  # benchmark/

        for dataset in config.datasets:
            # Make dataset_path absolute if it is a local path, otherwise keep it as is
            dataset_path = Path(dataset.dataset_path)
            if not dataset.is_remote and not dataset_path.is_absolute():
                dataset_path = base_dir / dataset_path

            self.datasets.append(
                {"dataset": load_dataset(str(dataset_path), trust_remote_code=True, download_mode=download_mode),
                "dataset_path": str(dataset_path),
                "dataset_name": dataset.__class__.__name__,
                 "dataset_metrics": dataset.metrics,
                "is_remote": dataset.is_remote})

    def evaluate(self):
        """
        Runs evaluation on all loaded datasets.

        Iterates over each test split, uses the predictor agent to generate answers,
        and compares them to the target values.

        Prints:
            - A tqdm loading bar while evaluating
            - Final evaluation results showing prediction vs target
        """
        results = {"pred": [], "ground_truth": []}

        for dataset in self.datasets:
            for example in tqdm(dataset["dataset"]["test"],
                                desc=f"Evaluating examples from {dataset['dataset_name']} dataset"):
                path_obj = Path(example["context"]["csv"])

                if dataset["is_remote"]:
                    from huggingface_hub import hf_hub_download
                    full_path = Path(
                        hf_hub_download(repo_id=dataset["dataset_path"], repo_type="dataset", filename=str(path_obj)))
                else:
                    current_file_path = Path(__file__).resolve().parent
                    base_path = current_file_path / ".." / Path(dataset["dataset_path"]).parent
                    full_path = base_path / path_obj

                additional_prompt = "There is a Question provided with a related table. Answer the question with a target value. "

                pred = self.predictor.eval(question=additional_prompt + example["utterance"], dataset=full_path,
                                           additional_info=[])

                results["pred"].append(pred)
                results["ground_truth"].append(example["target_value"])

                if self.verbose:
                    print("Question:", example["utterance"])
                    print(f"Result: {pred} -- {example['target_value']}")

            self.calculate_metrics(results, dataset["dataset_name"], dataset["dataset_metrics"])

    def calculate_metrics(self, results: dict, dataset_name: str, dataset_metrics: List[Metric]):
        print(f"Evaluation results for dataset {dataset_name}:")
        for metric in dataset_metrics:
            score = metric.compute(predictions=results["pred"], references=results["ground_truth"])
            print(f"{metric.metric_name}: {score}")
