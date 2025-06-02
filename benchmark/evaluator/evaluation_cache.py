import json
from pathlib import Path
from typing import List

from benchmark.evaluator.eval_config import EvalConfig

class Run:
    def __init__(self, run_id: int, is_finished: bool = False):
        self.run_id = run_id
        self.is_finished = is_finished

    @classmethod
    def from_dict(cls, d):
        return cls(run_id=d["run_id"], is_finished=d.get("is_finished", False))

    def to_dict(self):
        return {"run_id": self.run_id, "is_finished": self.is_finished}


class Example:
    def __init__(self, index: int, pred: str, ground_truth: str):
        self.index = index
        self.pred = pred
        self.ground_truth = ground_truth

    @classmethod
    def from_dict(cls, d):
        return cls(index=d["index"], pred=d["pred"], ground_truth=d["ground_truth"])

    def to_dict(self):
        return {"index": self.index, "pred": self.pred, "ground_truth": self.ground_truth}


class EvaluationCache:
    def __init__(self, config: EvalConfig):
        self.run = Run(0)
        self.config = config

        # Create general cache dir and run.json to store the current status of the runs
        self.cache_path = Path(__file__).resolve().parent / ".cache"
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.runs_file = self.cache_path / "runs.json"

        self.is_new_run = self.load_current_run()
        self.run_path = self.cache_path / f"run_{self.run.run_id}"

        if self.is_new_run:
            for dataset in self.config.datasets:
                self.run_path.mkdir(parents=True, exist_ok=False)
                example_file = self.run_path / f"examples_{dataset.__class__.__name__}.json"
                example_file.write_text(json.dumps([]))

    def load_current_run(self) -> bool:
        if not self.runs_file.exists():
            self.runs_file.write_text(json.dumps([]))

        with self.runs_file.open("r", encoding="utf-8") as f:
            run_dicts = json.load(f)

        run_objects = [Run.from_dict(d) for d in run_dicts]

        # Check if the last run is not finished
        if run_objects and not run_objects[-1].is_finished:
            answer = input("Last run is not finished. Do you want to continue it? (y/n): ").strip().lower()

            if answer == "y":
                is_new_run = False
                self.run = run_objects[-1]
                print(f"Continuing run: {self.run.run_id}")
            else:
                is_new_run = True
                self.run = Run(run_id=run_objects[-1].run_id + 1)
                run_objects.append(self.run)
                print(f"Starting new run: {self.run.run_id}")
        else:
            is_new_run = True
            if run_objects:
                self.run = Run(run_id=run_objects[-1].run_id + 1)
            else:
                self.run = Run(run_id=0)
            run_objects.append(self.run)
            print(f"Starting new run: {self.run.run_id}")

        with self.runs_file.open("w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in run_objects], f, indent=2)

        return is_new_run

    def safe_example(self, index: int, pred: str, ground_truth: str, dataset_name: str):
        new_example = Example(index, pred, ground_truth)
        example_file = self.run_path / f"examples_{dataset_name}.json"

        with example_file.open("r", encoding="utf-8") as f:
            example_dicts = json.load(f)

        examples = [Example.from_dict(d) for d in example_dicts]

        examples.append(new_example)

        with example_file.open("w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in examples], f, indent=2)

    def finish_run(self):
        with self.runs_file.open("r", encoding="utf-8") as f:
            run_dicts = json.load(f)

        run_objects = [Run.from_dict(d) for d in run_dicts]

        run_objects[-1].is_finished = True

        with self.runs_file.open("w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in run_objects], f, indent=2)

    def get_cached_results(self, dataset_name: str) -> (dict, List[int]):
        if self.is_new_run:
            return {"pred": [], "ground_truth": []}, []
        else:
            example_file = self.run_path / f"examples_{dataset_name}.json"
            with example_file.open("r", encoding="utf-8") as f:
                example_dicts = json.load(f)

            examples = [Example.from_dict(d) for d in example_dicts]

            results = {"pred": [], "ground_truth": []}
            indices = []
            for example in examples:
                results["pred"].append(example.pred)
                results["ground_truth"].append(example.ground_truth)
                indices.append(example.index)

            return results, indices
