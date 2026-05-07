#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -x "$DIR/.venv/bin/python3" ]; then
  PYTHON="$DIR/.venv/bin/python3"
else
  PYTHON="${PYTHON:-python3}"
fi

"$PYTHON" "$DIR/detector.py" "$@"
