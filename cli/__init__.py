"""Shared conversion engine behind the ``bids_convert.py`` entry point.

Both modalities (scalp and intracranial) run through the same job builder,
stage-gating, overwrite resolution, orchestrator and validation layer. The
only modality-specific code left is the converter classes themselves, which
the registry resolves per experiment.

Importing this package puts the repository root on ``sys.path`` so the
top-level helper modules (``conversion_error_log``, ``bids_validation``) and
the ``intracranial`` / ``scalp`` converter packages resolve the same way in
the driver process and on Dask workers.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
