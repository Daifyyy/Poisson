import argparse
from pathlib import Path

from utils.poisson_utils.data import load_data


def convert_path(path: Path) -> None:
    """Convert CSV files to Parquet using :func:`load_data`.

    ``load_data`` automatically writes a Parquet file next to the CSV when it
    reads one. This helper simply calls it for each provided path.
    """
    if path.is_dir():
        for csv_file in path.glob("*.csv"):
            load_data(str(csv_file))
    elif path.suffix.lower() == ".csv" and path.exists():
        load_data(str(path))
    else:
        print(f"Skipping {path}: not a CSV file or directory", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preconvert CSV files to Parquet for faster future loads"
    )
    parser.add_argument(
        "paths", nargs="+", help="CSV files or directories containing CSVs"
    )
    args = parser.parse_args()

    for p in args.paths:
        convert_path(Path(p))


if __name__ == "__main__":
    main()
