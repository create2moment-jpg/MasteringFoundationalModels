"""Environment preflight: what can this machine run, and what would a lab download?

Read-only by design: it imports nothing heavy unless present, never downloads,
never writes outside an explicitly requested output path, and network probes
are opt-in.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Packages that matter for this curriculum, in rough stack order.
TRACKED_PACKAGES: tuple[str, ...] = (
    "numpy", "pydantic", "PyYAML",
    "torch", "torchvision", "transformers", "timm", "accelerate", "peft",
    "onnx", "onnxruntime", "onnxruntime-gpu", "tensorrt", "torch-tensorrt", "openvino",
    "mmengine", "mmcv", "mmdet", "mmdet3d", "pcdet", "spconv-cu120", "spconv-cu121", "spconv-cu124",
    "ultralytics", "nuscenes-devkit", "pycocotools",
)  # fmt: skip

NETWORK_PROBES: tuple[str, ...] = (
    "https://huggingface.co/api/models/facebook/dinov2-base",
    "https://github.com",
    "https://pypi.org/simple/numpy/",
)


def package_versions(names: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for n in names:
        try:
            out[n] = importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:
            out[n] = None
    return out


@dataclass
class GpuInfo:
    torch_cuda_available: bool | None = None
    torch_cuda_version: str | None = None
    devices: list[str] = field(default_factory=list)
    nvidia_smi: str | None = None


def gpu_info() -> GpuInfo:
    info = GpuInfo()
    if importlib.util.find_spec("torch") is not None:
        import torch

        info.torch_cuda_available = bool(torch.cuda.is_available())
        info.torch_cuda_version = getattr(torch.version, "cuda", None)
        if info.torch_cuda_available:
            info.devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10, check=False,
            )  # fmt: skip
            info.nvidia_smi = r.stdout.strip() or r.stderr.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            info.nvidia_smi = f"error: {exc}"
    return info


def _mem_total_gb() -> float | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 2**20, 1)
    except OSError:
        return None
    return None


def probe_network(urls: tuple[str, ...] = NETWORK_PROBES, timeout: float = 8.0) -> dict[str, str]:
    """HEAD-request a few hosts; returns ``url -> "ok <status>" | "blocked: <reason>"``."""
    results: dict[str, str] = {}
    for url in urls:
        req = urllib.request.Request(url, method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                results[url] = f"ok {resp.status}"
        except urllib.error.HTTPError as exc:
            results[url] = f"http {exc.code}"
        except (urllib.error.URLError, OSError) as exc:
            results[url] = f"blocked: {getattr(exc, 'reason', exc)}"
    return results


@dataclass
class EnvReport:
    python: str
    executable_in_venv: bool
    platform: str
    cpu_count: int | None
    mem_total_gb: float | None
    disk_free_gb: float
    packages: dict[str, str | None]
    gpu: GpuInfo
    network: dict[str, str] | None
    capabilities: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def capabilities(pkgs: dict[str, str | None], gpu: GpuInfo) -> dict[str, bool]:
    """Coarse "can I run lab X?" flags derived from installed packages and devices."""
    has = {k: v is not None for k, v in pkgs.items()}
    return {
        "registry_and_cpu_labs": True,
        "torch": has["torch"],
        "cuda": bool(gpu.torch_cuda_available),
        "hf_inference": has["torch"] and has["transformers"],
        "onnx_export_and_parity": has["torch"]
        and has["onnx"]
        and (has["onnxruntime"] or has["onnxruntime-gpu"]),
        "tensorrt": has["tensorrt"],
        "openvino": has["openvino"],
        "openpcdet": has["pcdet"],
        "mmdetection3d": has["mmdet3d"],
        "sparse_conv": any(has[k] for k in pkgs if k.startswith("spconv")),
    }


def collect(network: bool = False, path: Path = Path(".")) -> EnvReport:
    pkgs = package_versions()
    gpu = gpu_info()
    return EnvReport(
        python=platform.python_version(),
        executable_in_venv=sys.prefix != sys.base_prefix or "VIRTUAL_ENV" in os.environ,
        platform=f"{platform.system()} {platform.release()} {platform.machine()}",
        cpu_count=os.cpu_count(),
        mem_total_gb=_mem_total_gb(),
        disk_free_gb=round(shutil.disk_usage(path).free / 2**30, 1),
        packages=pkgs,
        gpu=gpu,
        network=probe_network() if network else None,
        capabilities=capabilities(pkgs, gpu),
    )
