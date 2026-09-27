#!/bin/sh
# A generated hash manifest is output, not an input to its next generation.
set -eu
cd "$(dirname "$0")/../.."
.tools/conda-env/bin/python -B - <<'PY'
import os
from pathlib import Path
import shutil
import tempfile
from drills.experiment import load_config

config = load_config('experiments/fixed-state-normalization/protocol.yml')
manifest = Path(config['runtime']['output_dir']) / 'SHA256.json'
if manifest.exists():
    descriptor, backup = tempfile.mkstemp(prefix='drills-fixed-normalization-manifest-', suffix='.json')
    os.close(descriptor)
    shutil.move(str(manifest), backup)
    print('Previous generated manifest preserved at', backup)
PY
exec .tools/conda-env/bin/python -B experiments/fixed-state-normalization/run.py package
