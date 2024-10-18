import time
from contextlib import contextmanager
from typing import Callable


@contextmanager
def time_it():
    start_time = time.time()
    yield  # This allows the code block to run
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time taken: {elapsed_time:.2f} seconds")


def time__it(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Execution time: {end - start:.2f} seconds")
        return result

    return wrapper