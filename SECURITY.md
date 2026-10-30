# Security and data-handling policy

## Reporting

Report vulnerabilities privately to the repository owner via GitHub's private security advisory
feature, not in public issues.

## Model and code supply chain

- **`trust_remote_code=True`** (e.g. C-RADIO) executes code from the model repository. Pin a revision (commit SHA) and review the code before running it.
- Prefer `safetensors` over pickle-based `.pth` / `.pt` files. Load untrusted `.pth` only with `torch.load(..., weights_only=True)`.
- Pin model revisions and record file hashes. A `main` branch can change under you.
- Custom CUDA ops and TensorRT plugins from third-party repos are native code. Build them from pinned sources.

## Secrets and data

- Never commit HF tokens, cloud credentials, dataset download links with embedded keys, or `.env` files (they are git-ignored).
- Never commit proprietary sensor data, calibration files from production vehicles, or derived engines.
- Benchmark records are designed to exclude hostnames, usernames and absolute paths. Keep it that way.
