"""Deterministic seeding across Python, NumPy and (if installed) PyTorch."""

from __future__ import annotations

import importlib.util
import os
import random

import numpy as np


def seed_everything(seed: int, deterministic_torch: bool = False) -> None:
    """Seed all RNGs the labs use.

    ``deterministic_torch=True`` also requests deterministic cuDNN/cuBLAS kernels.
    That is slower and some ops (e.g. scatter-add used by voxel/pillar scatter,
    some sparse-conv kernels) have *no* deterministic CUDA implementation, in
    which case PyTorch raises – which is the point: you learn where
    nondeterminism enters your pipeline.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - legacy global RNG still used by many 3rd-party pipelines
    os.environ["PYTHONHASHSEED"] = str(seed)
    if importlib.util.find_spec("torch") is None:
        return
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
