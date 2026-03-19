"""
Compatibility shim.

The core aggregation/regex logic lives in `ipms_portal.data_processing`.
This file re-exports the public functions so the project layout matches the expected structure.
"""

from ipms_portal.data_processing import (  # noqa: F401
    ExperimentMeta,
    add_baf_core_indicator,
    extract_metadata_from_filename,
    load_and_aggregate_csv,
    scan_csv_files,
)

