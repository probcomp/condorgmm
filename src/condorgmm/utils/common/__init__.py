from pathlib import Path
import condorgmm
import subprocess as sp
import os

# Re-exports
from .image_utils import *  # noqa:F403
from .rerun import *  # noqa:F403
from .gmm import *  # noqa:F403
from .pose import *  # noqa:F403
from .camera import *  # noqa:F403
from .mesh import *  # noqa:F403


def get_root_path() -> Path:
    return Path(Path(condorgmm.__file__).parents[2])


def maybe_mkdir(path: Path | str) -> Path:
    if isinstance(path, str):
        path = Path(path)
    path = get_root_path() / path
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory {path}")
    return path


def get_assets() -> Path:
    assets_dir_path = get_root_path() / "assets"

    if not os.path.exists(assets_dir_path):
        os.makedirs(assets_dir_path)
        print(
            f"Initialized empty directory for shared bucket data at {assets_dir_path}."
        )

    return assets_dir_path


get_assets_path = get_assets


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = (
        sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    )
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values
