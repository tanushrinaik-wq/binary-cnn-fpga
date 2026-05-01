#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_DIR="$ROOT_DIR/python"
ARTIFACTS="$ROOT_DIR/artifacts"
DATA_ROOT="${1:-$ROOT_DIR/data}"
ZIP_OUT="$ROOT_DIR/deliverables_rtl_bundle.zip"

mkdir -p "$ARTIFACTS" "$ROOT_DIR/weights" "$ROOT_DIR/reports"

python -m pip install -r "$ROOT_DIR/requirements.txt"
python "$PY_DIR/train_teacher.py" --data-root "$DATA_ROOT" --out-dir "$ARTIFACTS" --epochs 3 --batch-size 64
python "$PY_DIR/train_student.py" --data-root "$DATA_ROOT" --artifacts "$ARTIFACTS" --epochs 12 --batch-size 64
python "$PY_DIR/binarize.py" --data-root "$DATA_ROOT" --artifacts "$ARTIFACTS" --epochs 6 --batch-size 64
python "$PY_DIR/export_weights.py" --data-root "$DATA_ROOT" --artifacts "$ARTIFACTS" --weights-out "$ROOT_DIR/weights"
python "$PY_DIR/compare.py" --data-root "$DATA_ROOT" --artifacts "$ARTIFACTS" --report-dir "$ROOT_DIR/reports"
python "$PY_DIR/package_outputs.py" --project-root "$ROOT_DIR" --output "$ZIP_OUT"
echo "Bundle written to $ZIP_OUT"
