import os
import jax


def trace(func, *args, **kwargs):
    trace_dir = os.path.join(os.getenv("PIXI_PROJECT_ROOT", "/tmp"), "tensorboard")
    with jax.profiler.trace(trace_dir):
        result = func(*args, **kwargs)
        jax.block_until_ready(result)
    return result
