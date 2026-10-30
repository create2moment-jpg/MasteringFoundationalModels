"""Load, validate, query and export the model registry."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from model_lab.registry.schema import ModelEntry, Registry, Role, Task

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY = REPO_ROOT / "models" / "registry.yaml"
DEFAULT_SCHEMA = REPO_ROOT / "models" / "registry.schema.json"


class RegistryError(Exception):
    """Raised when the registry file is missing, unparsable or fails validation."""


def load_registry(path: Path = DEFAULT_REGISTRY) -> Registry:
    """Parse and validate a registry YAML file.

    Raises:
        RegistryError: with a human-readable list of validation problems.
    """
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RegistryError(f"registry not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise RegistryError(f"invalid YAML in {path}: {exc}") from exc
    try:
        return Registry.model_validate(raw)
    except ValidationError as exc:
        lines = [f"{path}: {exc.error_count()} validation error(s)"]
        for err in exc.errors():
            loc = _describe_loc(raw, err["loc"])
            lines.append(f"  - {loc}: {err['msg']}")
        raise RegistryError("\n".join(lines)) from exc


def _describe_loc(raw: Any, loc: tuple[int | str, ...]) -> str:
    """Turn ``('models', 3, 'licensing')`` into ``models[3:rt-detrv2-r50].licensing``."""
    parts: list[str] = []
    node = raw
    for key in loc:
        if isinstance(key, int):
            label = str(key)
            if isinstance(node, list) and key < len(node) and isinstance(node[key], dict):
                label = f"{key}:{node[key].get('id', '?')}"
            parts[-1] = f"{parts[-1]}[{label}]" if parts else f"[{label}]"
            node = node[key] if isinstance(node, list) and key < len(node) else None
        else:
            parts.append(str(key))
            node = node.get(key) if isinstance(node, dict) else None
    return ".".join(parts) or "<root>"


def get_model(registry: Registry, model_id: str) -> ModelEntry:
    for model in registry.models:
        if model.id == model_id or model_id in model.aliases:
            return model
    raise KeyError(model_id)


def filter_models(
    models: Iterable[ModelEntry],
    *,
    task: Task | None = None,
    role: Role | None = None,
) -> list[ModelEntry]:
    out = []
    for m in models:
        if task is not None and task not in m.tasks:
            continue
        if role is not None and m.role is not role:
            continue
        out.append(m)
    return out


def json_schema() -> dict[str, Any]:
    """JSON Schema for editors / external validators (generated from the pydantic models)."""
    return Registry.model_json_schema()


def write_json_schema(path: Path = DEFAULT_SCHEMA) -> None:
    path.write_text(json.dumps(json_schema(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
