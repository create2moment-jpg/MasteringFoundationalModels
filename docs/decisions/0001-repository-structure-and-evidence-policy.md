# ADR 0001: Repository structure and evidence policy

- **Status:** accepted
- **Date:** 2026-10-30

## Context

The generating prompt asked for a repository named `foundation-model-mastery-lab/`. The work
happened inside an existing, empty Git repository (`MasteringFoundationalModels`). Facts about
models change monthly, and the build environment could reach the Hugging Face Hub (via a
connector) and web search, but not github.com, arxiv.org or vendor sites directly. It had
no GPU and no PyTorch.

## Decisions

1. **Build at the repository root** rather than in a nested `foundation-model-mastery-lab/` folder. The Python package is `model-lab` (import `model_lab`) as specified.
2. **Registry = YAML + pydantic schema** (strict: unknown keys rejected, enums for semantics), with a generated JSON Schema that CI checks for staleness.
3. **Evidence is first-class:** every entry has a `verification` block (status, method, evidence list with source type and access date). `verified_primary` / `partially_verified` require evidence. Prior-knowledge facts are labelled as such.
4. **Five artifact types are tracked separately** (pretrained, automotive fine-tune, quantised, ONNX, engine), each with an explicit status enum. URLs are required for "available" statuses.
5. **Core package stays light** (numpy, pydantic, pyyaml). Torch, HF, ONNX and LiDAR frameworks are extras or external, so CI and the CPU labs run anywhere.
6. **Reference implementations over wrappers** for geometry, voxelisation and metrics: slow but readable, unit-tested, and usable as parity oracles for CUDA versions.
7. **Synthetic LiDAR fixtures** replace datasets in tests. Results on them are never reported as model performance.
8. **Only create directories that have content**, so labs 02, 03, 06, 07, 08 and 10 are roadmap items, not empty folders.

## Consequences

- 19 of 23 registry entries sit in the verification queue (`model-lab registry audit`), which is honest but means the playbook often answers "no verified winner".
- GPU-dependent recipes ship as written-but-unexecuted scripts, clearly labelled.

## Template for future ADRs

Context → Options considered (with evidence links) → Decision → Consequences → Revisit-by date.
