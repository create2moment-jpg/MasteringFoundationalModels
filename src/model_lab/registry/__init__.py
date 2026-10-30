"""Model registry: typed schema, loader and queries."""

from model_lab.registry.loader import (
    DEFAULT_REGISTRY,
    RegistryError,
    filter_models,
    get_model,
    load_registry,
)
from model_lab.registry.schema import ArtifactStatus, ModelEntry, Registry, Role, Task

__all__ = [
    "DEFAULT_REGISTRY",
    "ArtifactStatus",
    "ModelEntry",
    "Registry",
    "RegistryError",
    "Role",
    "Task",
    "filter_models",
    "get_model",
    "load_registry",
]
