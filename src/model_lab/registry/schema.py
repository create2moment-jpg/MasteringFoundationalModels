"""Typed schema for the autonomous-driving perception model registry.

The registry (``models/registry.yaml``) is the single source of truth for model
references in this repo. The schema is intentionally strict about *semantics*
(enums for roles, tasks, artifact status, claim type) and permissive about
*missing data*: unknown values are ``None`` or an explicit ``not_verified``
status, never a guessed number.

Key distinctions enforced here (see docs/01_model_taxonomy.md):

* ``role`` separates foundation encoders from task detectors, fusion models, etc.
* ``Benchmark.claim_type`` separates paper/repo claims from local reproductions.
* ``Artifacts`` separates a training checkpoint from a quantized checkpoint,
  a portable export (ONNX) and a compiled engine (TensorRT plan).
"""

from __future__ import annotations

import datetime as dt
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator


class _Strict(BaseModel):
    """Base model: reject unknown keys so typos in YAML fail loudly."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)


class Role(str, Enum):
    """What the model *is*, independent of what it can be adapted to do."""

    FOUNDATION_ENCODER = "foundation_encoder"  # general-purpose pretrained features, no task head
    TASK_DETECTOR = "task_detector"  # trained end-to-end for detection on a fixed label set
    OPEN_VOCAB_DETECTOR = "open_vocab_detector"  # text-prompted / grounded detection
    SEGMENTER = "segmenter"
    FUSION_MODEL = "fusion_model"  # multi-sensor model with a task head
    TEMPORAL_DETECTOR = "temporal_detector"  # detector whose design centres on temporal fusion
    TRACKER = "tracker"


class Task(str, Enum):
    DET_2D = "2d_detection"
    OPEN_VOCAB_DET_2D = "open_vocabulary_2d_detection"
    SEG_2D = "2d_segmentation"
    IMAGE_FEATURES = "image_features"
    POINT_FEATURES = "point_cloud_features"
    DET_3D_LIDAR = "lidar_3d_detection"
    DET_3D_CAMERA = "camera_3d_detection"
    DET_3D_FUSION = "camera_lidar_3d_detection"
    SEG_3D = "3d_segmentation"
    BEV_MAP_SEG = "bev_map_segmentation"
    TRACKING_3D = "3d_tracking"


class Modality(str, Enum):
    CAMERA_MONO = "camera_mono"
    CAMERA_MULTI = "camera_multi_view"
    LIDAR = "lidar"
    TEXT = "text"
    RADAR = "radar"


class DomainFit(str, Enum):
    """How the *pretraining* data relates to outdoor driving scenes."""

    AUTOMOTIVE = "automotive"  # pretrained/trained on driving datasets
    MIXED_INCLUDES_AUTOMOTIVE = "mixed_includes_automotive"
    GENERIC_IMAGES = "generic_web_images"
    GENERIC_DETECTION = "generic_detection_coco_like"
    INDOOR_OR_OBJECT_3D = "indoor_or_object_3d"
    UNKNOWN = "unknown"


class CommercialUse(str, Enum):
    PERMISSIVE = "permissive"  # e.g. Apache-2.0 / MIT weights (dataset terms may still apply)
    CONDITIONAL = "conditional"  # custom licence that allows commercial use with conditions
    NON_COMMERCIAL = "non_commercial"
    COPYLEFT = "copyleft"  # e.g. AGPL – commercial use possible but with obligations
    UNCLEAR = "unclear_legal_review"


class ArtifactStatus(str, Enum):
    AVAILABLE_OFFICIAL = "available_official"
    AVAILABLE_COMMUNITY = "available_community"
    COMMUNITY_CONVERSION = "community_conversion"
    BUILD_RECIPE_AVAILABLE = "build_recipe_available"
    REQUIRES_LOCAL_BUILD = "requires_local_build"
    GATED = "gated"
    NOT_OFFERED = "not_offered"
    NOT_FOUND_AFTER_SEARCH = "not_found_after_search"
    NOT_VERIFIED = "not_verified"
    NOT_APPLICABLE = "not_applicable"


class ClaimType(str, Enum):
    REPORTED_PAPER = "reported_paper"
    REPORTED_REPO = "reported_repo"  # model zoo / model card table
    REPRODUCED_LOCAL = "reproduced_local"  # only with a benchmarks/results record


class VerificationStatus(str, Enum):
    VERIFIED_PRIMARY = "verified_primary"  # key facts checked against primary source
    PARTIALLY_VERIFIED = "partially_verified"  # some fields from primary, rest secondary/memory
    NOT_VERIFIED = "not_verified"  # recorded from prior knowledge; in verification queue


class SourceType(str, Enum):
    PAPER = "paper"
    OFFICIAL_REPO = "official_repo"
    MODEL_CARD = "model_card"
    OFFICIAL_DOCS = "official_docs"
    HUB_METADATA = "hub_metadata"  # e.g. HF API: license tag, params, gated flag
    SECONDARY = "secondary"  # aggregators, mirrors, search snippets
    PRIOR_KNOWLEDGE = "prior_knowledge"  # author knowledge, must be re-verified


class PeftSupport(str, Enum):
    SUPPORTED = "supported"  # tooling exists and is used in practice for this architecture
    EXPERIMENTAL = "experimental"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class Paper(_Strict):
    title: str
    year: int | None = None
    arxiv: str | None = Field(default=None, pattern=r"^\d{4}\.\d{4,5}$")
    url: HttpUrl | None = None


class Links(_Strict):
    project: HttpUrl | None = None
    code: HttpUrl | None = None
    model_card: HttpUrl | None = None
    weights: HttpUrl | None = None
    revision: str | None = None  # tag / commit / HF revision when pinned


class Licensing(_Strict):
    code_license: str | None = None
    weight_license: str | None = None
    dataset_terms: str | None = None
    commercial_use: CommercialUse = CommercialUse.UNCLEAR
    gated: bool | None = None
    notes: str | None = None


class IOContract(_Strict):
    framework: str | None = None
    input: str | None = None
    output: str | None = None
    coordinate_convention: str | None = None
    preprocessing: str | None = None
    postprocessing: str | None = None


class Pretraining(_Strict):
    data: str | None = None
    domain_fit: DomainFit = DomainFit.UNKNOWN
    notes: str | None = None


class Benchmark(_Strict):
    dataset: str
    split: str
    metric: str
    value: float | None = None
    protocol: str | None = None
    claim_type: ClaimType
    source: HttpUrl | None = None
    notes: str | None = None


class Size(_Strict):
    params_m: float | None = Field(default=None, gt=0)
    source: SourceType | None = None


class FineTuning(_Strict):
    recipe_available: bool | None = None
    peft: PeftSupport = PeftSupport.UNKNOWN
    notes: str | None = None


class ArtifactRef(_Strict):
    status: ArtifactStatus
    url: HttpUrl | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _url_required_when_available(self) -> ArtifactRef:
        needs_url = {
            ArtifactStatus.AVAILABLE_OFFICIAL,
            ArtifactStatus.AVAILABLE_COMMUNITY,
            ArtifactStatus.COMMUNITY_CONVERSION,
            ArtifactStatus.BUILD_RECIPE_AVAILABLE,
        }
        if self.status in needs_url and self.url is None:
            raise ValueError(f"artifact status '{self.status.value}' requires a url")
        return self


class Artifacts(_Strict):
    """The five artifact kinds the prompt requires to be kept distinct."""

    official_pretrained: ArtifactRef
    automotive_finetuned: ArtifactRef
    quantized: ArtifactRef
    onnx_export: ArtifactRef
    engine: ArtifactRef


class Deployment(_Strict):
    export_formats: list[str] = Field(default_factory=list)
    runtimes: list[str] = Field(default_factory=list)
    known_traps: list[str] = Field(default_factory=list)


class Evidence(_Strict):
    claim: str
    url: HttpUrl
    source_type: SourceType
    accessed: dt.date


class Verification(_Strict):
    last_verified: dt.date
    status: VerificationStatus
    method: str
    evidence: list[Evidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def _evidence_for_verified(self) -> Verification:
        if self.status is not VerificationStatus.NOT_VERIFIED and not self.evidence:
            raise ValueError("verified/partially_verified entries must list evidence")
        return self


class ModelEntry(_Strict):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_\-.]*$")
    name: str
    aliases: list[str] = Field(default_factory=list)
    family: str
    role: Role
    tasks: list[Task] = Field(min_length=1)
    modalities: list[Modality] = Field(min_length=1)
    open_vocabulary: bool = False
    summary: str
    paper: Paper | None = None
    links: Links
    licensing: Licensing
    io_contract: IOContract = Field(default_factory=IOContract)
    pretraining: Pretraining = Field(default_factory=Pretraining)
    benchmarks: list[Benchmark] = Field(default_factory=list)
    size: Size = Field(default_factory=Size)
    finetuning: FineTuning = Field(default_factory=FineTuning)
    artifacts: Artifacts
    deployment: Deployment = Field(default_factory=Deployment)
    recommendation_context: str | None = None
    caveats: list[str] = Field(default_factory=list)
    card: str | None = None  # relative path to models/cards/*.md
    verification: Verification

    @model_validator(mode="after")
    def _open_vocab_consistency(self) -> ModelEntry:
        if self.role is Role.OPEN_VOCAB_DETECTOR and not self.open_vocabulary:
            raise ValueError("role open_vocab_detector requires open_vocabulary: true")
        return self


class Registry(_Strict):
    schema_version: int = 1
    research_baseline: dt.date
    models: list[ModelEntry]

    @field_validator("models")
    @classmethod
    def _unique_ids(cls, models: list[ModelEntry]) -> list[ModelEntry]:
        seen: set[str] = set()
        dupes: set[str] = set()
        for m in models:
            (dupes if m.id in seen else seen).add(m.id)
        if dupes:
            raise ValueError(f"duplicate model ids: {sorted(dupes)}")
        return models
