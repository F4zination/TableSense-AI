import time
import tracemalloc
from functools import wraps

def measure_performance(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        tracemalloc.start()
        start_time = time.perf_counter()
        result = func(self, *args, **kwargs)
        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        usage_metadata = getattr(result, "response_metadata", None)
        input_tokens = output_tokens = None

        if usage_metadata:
            input_tokens = usage_metadata["token_usage"]["prompt_tokens"]
            output_tokens = usage_metadata["token_usage"]["completion_tokens"]

        print(f"--- {func.__name__} Metrics ---")
        print(f"Time elapsed: {end_time - start_time:.4f} seconds")
        print(f"Peak memory usage: {peak / 1024:.2f} KB")
        if input_tokens is not None:
            print(f"Input tokens: {input_tokens}")
        if output_tokens is not None:
            print(f"Output tokens: {output_tokens}")
        print(f"--------------------")

        return result
    return wrapper
