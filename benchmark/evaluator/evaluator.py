from pandas.errors import ParserError
from pathlib import Path
from typing import List

from datasets import load_dataset
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import pandas as pd

from benchmark.evaluator.evaluation_cache import EvaluationCache
from benchmark.evaluator.metrics.metric import Metric
from tablesense_ai.agent.base import BaseAgent
from benchmark.evaluator.eval_config import EvalConfig
from tablesense_ai.utils import canonicaliser


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
        self.cache = EvaluationCache(config)

        download_mode = (
            "force_redownload"
            if config.force_redownload
            else "reuse_dataset_if_exists"
        )

        current_file_path = Path(__file__).resolve().parent  # path to evaluator.py
        base_dir = current_file_path.parent  # benchmark/

        for dataset in config.datasets:
            if dataset.is_remote:
                # ensure remote paths use forward slashes
                dataset_path = Path(dataset.dataset_path).as_posix()
            else:
                dataset_path = Path(dataset.dataset_path)
                if not dataset_path.is_absolute():
                    dataset_path = base_dir / dataset_path

            self.datasets.append({
                "dataset": load_dataset(
                    str(dataset_path),
                    trust_remote_code=True,
                    download_mode=download_mode
                ),
                "dataset_path": str(dataset_path),
                "dataset_name": dataset.__class__.__name__,
                "dataset_metrics": dataset.metrics,
                "is_remote": dataset.is_remote,
                "system_prompt": dataset.system_prompt,
            })

    def _read_table_content(self, file_path: Path) -> str:
        """Read and convert table content to HTML representation."""
        # read CSV into DataFrame and return as HTML table
        try:
            # Use on_bad_lines='skip' to ignore rows with the wrong number of columns
            df = pd.read_csv(file_path, on_bad_lines='skip')
            return df.to_html(index=False, escape=False)

        except (ParserError, pd.errors.EmptyDataError) as e:
            # Log the specific file that failed and why
            print(f"Skipping file {file_path} due to error: {e}")
            # Return an empty table or error message so the loop can continue
            return "<p>Error reading table data.</p>"

        except Exception as e:
            # Catch any other unexpected errors
            print(f"An unexpected error occurred with file {file_path}: {e}")
            return "<p>Error reading table data.</p>"

    def evaluate(self):
        scores = []

        for dataset in self.datasets:
            results, evaluated_indices = self.cache.get_cached_results(dataset["dataset_name"])
            print(results)

            for index, example in enumerate(
                    tqdm(dataset["dataset"]["test"],
                         desc=f"Evaluating examples from {dataset['dataset_name']} dataset")
            ):
                if index in evaluated_indices:
                    continue

                path_obj = Path(example["context"]["csv"])

                if dataset["is_remote"]:
                    remote_file = path_obj.as_posix()
                    local_file = hf_hub_download(
                        repo_id=dataset["dataset_path"],
                        repo_type="dataset",
                        filename=remote_file
                    )
                    full_path = Path(local_file)
                else:
                    current_file_path = Path(__file__).resolve().parent
                    base_path = current_file_path.parent / Path(dataset["dataset_path"]).parent
                    full_path = base_path / path_obj

                # Lese den Tabelleninhalt
                table_content = self._read_table_content(full_path)

                pred = self.predictor.eval(
                    question=example["utterance"],
                    dataset=full_path,
                    dataset_prompt=dataset["system_prompt"]
                )

                if str(pred) != "skipped-too-long":
                    target_value = example["target_value"]
                    if dataset["dataset_name"] in {"TabMWPSelection", "WikiTableQuestionsSelection"}:
                        pred = canonicaliser.clean(str(pred))
                        target_value = example["target_value"].lower().strip()
                    results["pred"].append(str(pred))
                    results["ground_truth"].append(target_value)
                    self.cache.safe_example(
                        index,
                        str(pred),
                        target_value,
                        dataset["dataset_name"],
                        question=example["utterance"],
                        table=table_content  # Speichere den Tabelleninhalt statt des Pfades
                    )
                    if self.verbose:
                        print("\nQuestion:", example["utterance"])
                        print(f"Result: {pred} -- {target_value}")
                else:
                    if self.verbose:
                        print(
                            f"Skipped example – empty result for question: {example['utterance']}"
                        )
            scores.append(self.calculate_metrics(
                results,
                dataset["dataset_name"],
                dataset["dataset_metrics"]
            ))
        self.cache.finish_run()
        return scores

    def calculate_metrics(
            self,
            results: dict,
            dataset_name: str,
            dataset_metrics: List[Metric]
    ):
        print(f"Evaluation results for dataset {dataset_name}:")
        final_scores = {"dataset_name": dataset_name}
        for metric in dataset_metrics:
            score = metric.compute(
                predictions=results["pred"],
                references=results["ground_truth"]
            )
            print(f"{metric.metric_name}: {score}")
            final_scores[metric.metric_name] = score

        return final_scores
