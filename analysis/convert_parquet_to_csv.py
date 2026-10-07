"""Convert a simulation Parquet file to CSV."""

import argparse
from pathlib import Path

import pandas as pd


_DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "results" / "data"


def convert_parquet_to_csv(
    parquet_path: str | Path,
    output_directory: str | Path | None = None,
) -> Path:
    """Convert one Parquet file to a same-named CSV file.

    Args:
        parquet_path: Path to the source Parquet file.
        output_directory: Optional output directory. Defaults to
            ``results/data`` in the project root.

    Returns:
        The path of the written CSV file.
    """
    source = Path(parquet_path)
    if source.suffix.lower() != ".parquet":
        raise ValueError("parquet_path must point to a .parquet file.")
    if not source.is_file():
        raise FileNotFoundError(f"Parquet file does not exist: {source}")

    destination_directory = (
        Path(output_directory) if output_directory is not None else _DATA_DIRECTORY
    )
    destination_directory.mkdir(parents=True, exist_ok=True)
    destination = destination_directory / f"{source.stem}.csv"
    pd.read_parquet(source).to_csv(destination, index=False)
    return destination


def main() -> None:
    """Convert a file supplied as an argument or entered interactively."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet_file", nargs="?", help="Source Parquet file")
    args = parser.parse_args()
    parquet_file = args.parquet_file or input("Parquet file to convert: ").strip()
    destination = convert_parquet_to_csv(parquet_file)
    print(f"CSV file written to {destination}")


if __name__ == "__main__":
    main()
