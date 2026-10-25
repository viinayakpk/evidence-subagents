#!/usr/bin/env bash
# Write tests/fixtures/gym/grading_python_modules.txt: the top-level modules the grading Python
# (/opt/venv/bin/python3, run from /workspace like the evaluation) can import in the agent image.
# Free and offline: runs the image with no network. Used by task_rules.grader_missing_modules.
set -euo pipefail
image="${1:-evcarry/toolathlon-gym:ed735ba}"
out="$(dirname "$0")/../tests/fixtures/gym/grading_python_modules.txt"
id="$(docker image inspect "$image" --format '{{.Id}}')"
version="$(docker run --rm --network none --entrypoint /opt/venv/bin/python3 "$image" -c 'import sys; print(sys.version.split()[0])')"
{
  echo "# Top-level modules the grading Python can import: /opt/venv/bin/python3 (Python ${version}) run from /workspace"
  echo "# in ${image} (image ${id:0:19}), $(date +%F). Stdlib, builtins, site-packages and the"
  echo "# gym's top-level directories (namespace packages such as tasks). Regenerate with scripts/grading_modules.sh."
  docker run --rm --network none -w /workspace --entrypoint /opt/venv/bin/python3 "$image" -c "
import os, pkgutil, sys
names = set(sys.stdlib_module_names) | set(sys.builtin_module_names) | {m.name for m in pkgutil.iter_modules()}
names |= {d for d in os.listdir('.') if os.path.isdir(d) and d.isidentifier()}
print('\n'.join(sorted(names)))"
} > "$out"
echo "wrote $out"
