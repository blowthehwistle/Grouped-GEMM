#!/bin/bash
# 여러 config로 run_all.py를 순차 실행, config별로 결과 저장
# Usage: ./run_multi_config.sh [config_dir]

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_DIR="${1:-$SCRIPT_DIR/configs}"
RESULTS_BASE="$REPO_ROOT/results/benchmark"

cd "$REPO_ROOT"

if [ ! -d "$CONFIG_DIR" ]; then
  echo "Config dir not found: $CONFIG_DIR"
  exit 1
fi

for f in "$CONFIG_DIR"/*.yaml; do
  [ -f "$f" ] || continue
  name=$(basename "$f" .yaml)
  echo ""
  echo "========================================"
  echo "Config: $name"
  echo "========================================"
  python "$SCRIPT_DIR/run_all.py" --config "$f" --results-subdir "$name"
done

echo ""
echo "Done. Results under: $RESULTS_BASE/"
ls -la "$RESULTS_BASE" 2>/dev/null || true
