#!/usr/bin/env bash
# Run every CPU check CI runs. Usage: scripts/check.sh   (from the repo root, venv active)
set -euo pipefail
ruff check .
ruff format --check .
mypy
model-lab registry validate --check-schema-file
pytest -q -m "not integration and not gpu"
python labs/04_point_cloud_basics/explore_scene.py --out outputs/lab04 > /dev/null
python labs/05_lidar_3d_baseline/run_cluster_baseline.py --scenes 5 > /dev/null
if python -c "import onnxruntime" 2>/dev/null; then
  python labs/09_quantization_and_export/onnx_parity_toy.py --out outputs/lab09 > /dev/null
else
  echo "skip lab 09 (pip install -e '.[export]')"
fi
echo "all checks passed"
