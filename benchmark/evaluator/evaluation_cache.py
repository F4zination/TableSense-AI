import json
from pathlib import Path
from typing import List
import pandas as pd
from html import escape

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
    def __init__(self, index: int, pred: str, ground_truth: str, question: str | None = None, table: str | None = None):
        self.index = index
        self.pred = pred
        self.ground_truth = ground_truth
        # Optional extended fields for better traceability
        self.question = question
        self.table = table

    @classmethod
    def from_dict(cls, d):
        # Backward-compatible: old cache files may not have question/table
        return cls(
            index=d["index"],
            pred=d["pred"],
            ground_truth=d["ground_truth"],
            question=d.get("question"),
            table=d.get("table"),
        )

    def to_dict(self):
        out = {"index": self.index, "pred": self.pred, "ground_truth": self.ground_truth}
        if self.question is not None:
            out["question"] = self.question
        if self.table is not None:
            out["table"] = self.table
        return out


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
                self.run_path.mkdir(parents=True, exist_ok=True)
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

    def safe_example(self, index: int, pred: str, ground_truth: str, dataset_name: str, question: str | None = None, table: str | None = None):
        # Append to JSON cache (backwards compatible)
        new_example = Example(index, pred, ground_truth, question=question, table=table)
        example_file = self.run_path / f"examples_{dataset_name}.json"

        with example_file.open("r", encoding="utf-8") as f:
            example_dicts = json.load(f)

        examples = [Example.from_dict(d) for d in example_dicts]

        examples.append(new_example)

        with example_file.open("w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in examples], f, indent=2)

        # Also create/update a human-friendly HTML cache file that shows tables as real tables
        try:
            self._write_html_cache(dataset_name, examples)
        except Exception as e:
            # Don't fail the run if HTML cache fails; keep JSON primary
            print(f"Warning: failed to write HTML cache for {dataset_name}: {e}")

    def _render_table_html(self, table_content: str) -> str:
        """Convert various table representations into an HTML table string.
        Supports:
        - HTML table fragments (returned as-is)
        - JSON table strings (pd.read_json)
        - CSV file paths (reads file)
        - raw CSV content
        - fallback: escaped preformatted text
        """
        if not table_content:
            return "<em>No table provided</em>"

        s = table_content.strip()

        # If already an HTML table fragment
        if s.startswith("<table") or s.startswith("<!doctype") or s.startswith("<html"):
            return table_content

        # JSON-like records
        try:
            if s[0] in "[{":
                df = pd.read_json(s)
                return df.to_html(index=False, escape=False)
        except Exception:
            pass

        # If it's a path to a file
        try:
            p = Path(table_content)
            if p.exists():
                # try csv
                try:
                    df = pd.read_csv(p)
                    return df.to_html(index=False, escape=False)
                except Exception:
                    # try json
                    try:
                        df = pd.read_json(p)
                        return df.to_html(index=False, escape=False)
                    except Exception:
                        pass
        except Exception:
            pass

        # If it looks like CSV content (contains newlines and commas)
        if "\n" in s and ("," in s or "\t" in s):
            try:
                from io import StringIO
                sep = "\t" if "\t" in s and s.count('\t') > s.count(',') else ","
                df = pd.read_csv(StringIO(s), sep=sep)
                return df.to_html(index=False, escape=False)
            except Exception:
                pass

        # Fallback: show escaped text
        return f"<pre>{escape(table_content)}</pre>"

    def _write_html_cache(self, dataset_name: str, examples: List[Example]):
        html_file = self.run_path / f"examples_{dataset_name}.html"

        parts = []
        parts.append("<!doctype html>")
        parts.append("<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">")
        parts.append("<title>Examples - %s</title>" % escape(dataset_name))
        # basic spreadsheet style
        parts.append("<style>body{font-family:Arial,Helvetica,sans-serif;padding:10px}table{border-collapse:collapse;margin:8px 0}table,th,td{border:1px solid #bbb;padding:6px}th{background:#f2f2f2}pre{white-space:pre-wrap;background:#fafafa;padding:8px;border:1px solid #eee}</style>")
        parts.append("</head><body>")
        parts.append(f"<h1>Cached examples for {escape(dataset_name)}</h1>")

        for ex in examples:
            parts.append(f"<section style=\"margin-bottom:18px\">")
            parts.append(f"<h2>Example {ex.index}</h2>")
            if ex.question:
                parts.append(f"<div><strong>Question:</strong> {escape(ex.question)}</div>")
            parts.append(f"<div><strong>Prediction:</strong> {escape(str(ex.pred))} &nbsp;&nbsp; <strong>Ground truth:</strong> {escape(str(ex.ground_truth))}</div>")
            parts.append("<div style=\"margin-top:8px\">")
            # Render the table content as HTML
            table_html = self._render_table_html(ex.table) if ex.table is not None else "<em>No table</em>"
            parts.append(table_html)
            parts.append("</div>")
            parts.append("</section>")

        parts.append("</body></html>")

        html_file.write_text("\n".join(parts), encoding="utf-8")

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
