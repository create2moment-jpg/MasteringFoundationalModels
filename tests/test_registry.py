import json
from pathlib import Path

import pytest
import yaml

from model_lab.registry.loader import (
    DEFAULT_REGISTRY,
    DEFAULT_SCHEMA,
    RegistryError,
    filter_models,
    get_model,
    json_schema,
    load_registry,
)
from model_lab.registry.schema import (
    ArtifactRef,
    ArtifactStatus,
    ClaimType,
    Role,
    Task,
    VerificationStatus,
)


@pytest.fixture(scope="module")
def raw() -> dict:
    return yaml.safe_load(DEFAULT_REGISTRY.read_text())


def test_shipped_registry_validates() -> None:
    reg = load_registry()
    assert len(reg.models) >= 15


def test_shipped_cards_exist() -> None:
    reg = load_registry()
    for m in reg.models:
        if m.card:
            assert (DEFAULT_REGISTRY.parent / m.card).is_file(), m.id


def test_committed_json_schema_is_current() -> None:
    assert json.loads(DEFAULT_SCHEMA.read_text()) == json_schema()


def test_no_local_reproduction_claims_without_results() -> None:
    """A reproduced_local benchmark must point at a committed result record."""
    results = Path(DEFAULT_REGISTRY).parents[1] / "benchmarks" / "results"
    for m in load_registry().models:
        for b in m.benchmarks:
            if b.claim_type is ClaimType.REPRODUCED_LOCAL:
                assert b.notes and (results / b.notes).is_file(), m.id


def test_verified_entries_carry_evidence() -> None:
    for m in load_registry().models:
        if m.verification.status is not VerificationStatus.NOT_VERIFIED:
            assert m.verification.evidence, m.id


def test_scope_covers_required_categories() -> None:
    reg = load_registry()
    for task in (Task.DET_2D, Task.IMAGE_FEATURES, Task.POINT_FEATURES, Task.DET_3D_LIDAR,
                 Task.DET_3D_CAMERA, Task.DET_3D_FUSION):  # fmt: skip
        assert filter_models(reg.models, task=task), task


def test_lookup_by_alias() -> None:
    reg = load_registry()
    assert get_model(reg, "facebook/dinov2-base").id == "dinov2-vitb14"
    with pytest.raises(KeyError):
        get_model(reg, "does-not-exist")


def test_filter_by_role() -> None:
    enc = filter_models(load_registry().models, role=Role.FOUNDATION_ENCODER)
    assert all(m.role is Role.FOUNDATION_ENCODER for m in enc)


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "reg.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def test_duplicate_ids_rejected(tmp_path: Path, raw: dict) -> None:
    data = {**raw, "models": [raw["models"][0], raw["models"][0]]}
    with pytest.raises(RegistryError, match="duplicate model ids"):
        load_registry(_write(tmp_path, data))


def test_unknown_field_rejected_with_located_message(tmp_path: Path, raw: dict) -> None:
    bad = {**raw["models"][0], "bogus_field": 1}
    with pytest.raises(RegistryError, match=r"models\[0:dinov2-vitb14\]\.bogus_field"):
        load_registry(_write(tmp_path, {**raw, "models": [bad]}))


def test_bad_enum_rejected(tmp_path: Path, raw: dict) -> None:
    bad = {**raw["models"][0], "role": "best_model_ever"}
    with pytest.raises(RegistryError, match="role"):
        load_registry(_write(tmp_path, {**raw, "models": [bad]}))


def test_verified_without_evidence_rejected(tmp_path: Path, raw: dict) -> None:
    bad = {**raw["models"][0]}
    bad["verification"] = {**bad["verification"], "evidence": []}
    with pytest.raises(RegistryError, match="evidence"):
        load_registry(_write(tmp_path, {**raw, "models": [bad]}))


def test_available_artifact_requires_url() -> None:
    with pytest.raises(ValueError, match="requires a url"):
        ArtifactRef(status=ArtifactStatus.AVAILABLE_OFFICIAL)
    ArtifactRef(status=ArtifactStatus.NOT_OFFERED)  # fine without url


def test_missing_file_message(tmp_path: Path) -> None:
    with pytest.raises(RegistryError, match="not found"):
        load_registry(tmp_path / "nope.yaml")
