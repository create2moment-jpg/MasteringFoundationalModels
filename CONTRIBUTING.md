# Contributing

## Development setup

```bash
uv venv && uv pip install -e ".[dev,export]"
pre-commit install            # optional: ruff + basic hygiene on commit
pytest -q && ruff check . && ruff format --check . && mypy
```

## Research rules

1. **Primary sources first:** paper, official repo (pinned tag or commit), official model card, official licence, official export docs. Community conversions are allowed but must be labelled (`available_community` / `community_conversion`).
2. **Separate claims:** paper (`reported_paper`), model zoo or card (`reported_repo`), your run (`reproduced_local` + result file).
3. **Separate licences:** code, weights, dataset terms. If unsure, use `commercial_use: unclear_legal_review`.
4. **Separate artifacts:** checkpoint, quantised weights, ONNX, engine.
5. **Driving relevance:** evaluate on driving data with range and size stratification. Generic benchmarks are context only.

## Updating model references

1. Verify the fact at its primary source and note the access date.
2. Edit `models/registry.yaml`:
   - change the field(s);
   - update `verification.last_verified`, `status`, `method`;
   - add an `evidence` item `{claim, url, source_type, accessed}`.
3. If a card exists, update its header date and §12 Sources.
4. Remove the item from the verification queue in `docs/sources/README.md`.
5. Run `model-lab registry validate --check-schema-file` and `pytest`.

## Adding a model

Add it only if it serves driving perception and you can fill licence, weights location and
I/O contract from primary sources. Fewer accurate entries beat a padded catalogue. Add a card
when the model becomes a recommended default.

## Adding a lab

Each lab needs: objective, prerequisites, commands, expected output (as actually observed, with
date and environment), "why it works / when it fails", quiz, stretch exercise, and cleanup
notes. GPU-heavy labs need a CPU or reduced-scale alternative.
