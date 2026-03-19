from __future__ import annotations

import argparse
import os

from app import LOCAL_DATA_DIR, DATA_SOURCE_DIR_DEFAULT, sync_csvs_to_local_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync IP-MS CSVs into local Data/ folder.")
    parser.add_argument("--source", default=DATA_SOURCE_DIR_DEFAULT, help="Source CSV directory.")
    parser.add_argument("--dest", default=LOCAL_DATA_DIR, help="Destination directory (default: ./Data).")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in destination.",
    )
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    counts = sync_csvs_to_local_data(args.source, args.dest, overwrite=args.overwrite)
    print(f"Sync complete: copied {counts['copied']} file(s), skipped {counts['skipped']} file(s).")


if __name__ == "__main__":
    main()

