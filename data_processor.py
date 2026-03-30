"""
Compatibility shim. Core logic lives in `ipms_portal`.
"""

from ipms_portal.biological_aliases import (  # noqa: F401
    expand_biological_target_string,
    infer_bait_gene_from_label,
    resolve_biological_fields,
)
from ipms_portal.data_processing import (  # noqa: F401
    ExperimentMeta,
    add_baf_core_indicator,
    add_experiment_biological_columns,
    enrich_meta_dict,
    extract_metadata_from_filename,
    load_and_aggregate_csv,
    parse_filename_fuzzy,
    scan_csv_files,
)
