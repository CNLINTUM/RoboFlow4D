"""
Backward-compatible wrapper for the old metric-debug inference entry point.

Use predict_flow_hdf5.py for new commands.
"""

from __future__ import annotations

import warnings

from predict_flow_hdf5 import main


if __name__ == "__main__":
    warnings.warn(
        "predict_flow_hdf5_metric_debug.py has been renamed to "
        "predict_flow_hdf5.py. Please update new commands to use the "
        "official entry point.",
        DeprecationWarning,
        stacklevel=1,
    )
    main()
