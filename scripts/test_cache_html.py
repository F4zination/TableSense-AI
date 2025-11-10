from benchmark.evaluator.eval_config import EvalConfig
from benchmark.evaluator.dataset_definition import DistilledTabMWP
from benchmark.evaluator.evaluation_cache import EvaluationCache

cfg = EvalConfig(datasets=[DistilledTabMWP()], force_redownload=False, verbose=False)
cache = EvaluationCache(cfg)

csv = "Name,Value\nAlice,10\nBob,20"
cache.safe_example(0, "30", "30", "DistilledTabMWP", question="Sum values", table=csv)

html_path = cache.run_path / "examples_DistilledTabMWP.html"
print('HTML exists:', html_path.exists())
print('HTML path:', str(html_path))

