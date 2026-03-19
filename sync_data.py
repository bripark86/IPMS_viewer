from __future__ import annotations

import argparse
import os
import shutil

LOCAL_DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")
DATA_SOURCE_DIR_DEFAULT = "/Users/sp1665/Downloads/IPMS/Janet_Liu"


def sync_csvs_to_local_data(source_dir: str, dest_dir: str, overwrite: bool = False) -> dict[str, int]:
    os.makedirs(dest_dir, exist_ok=True)
    copied = 0
    skipped = 0
    with os.scandir(source_dir) as src_it:
        for entry in src_it:
            if not entry.is_file() or not entry.name.lower().endswith(".csv"):
                continue
            dst_path = os.path.join(dest_dir, entry.name)
            if (not overwrite) and os.path.exists(dst_path):
                skipped += 1
                continue
            shutil.copy2(entry.path, dst_path)
            copied += 1
    return {"copied": copied, "skipped": skipped}


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

