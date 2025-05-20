from pathlib import Path
from typing import List

from datasets import load_dataset
from tqdm import tqdm

from agent import Agent
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

    def __init__(self, config: EvalConfig, agent: Agent):
        self.verbose = config.verbose
        self.predictor = agent
        self.datasets = []

        if config.force_redownload is True:
            download_mode = "force_redownload"
        else:
            download_mode = "reuse_dataset_if_exists"

        for dataset in config.datasets:
            self.datasets.append(
                {"dataset": load_dataset(dataset.dataset_path, trust_remote_code=True, download_mode=download_mode),
                 "dataset_path": dataset.dataset_path,
                 "dataset_name": dataset.__class__.__name__,
                 "is_remote": dataset.is_remote})

        self.metrics = config.metrics

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

            self.calculate_metrics(results)

    def calculate_metrics(self, results: dict):
        print("Evaluation results:")
        for metric in self.metrics:
            score = metric.compute(predictions=results["pred"], references=results["ground_truth"])
            print(f"{metric.metric_name}: {score}")
