"""Compare a planned drone path with the drone's actual Parquet data.

Usage:
    python compare_path_to_actual.py path.parquet actual.parquet
"""

import argparse

from compare_paths import compare_paths


def main() -> None:
    """Compare path and actual data supplied on the command line or by prompt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path_file", nargs="?", help="Planned path Parquet file")
    parser.add_argument("actual_file", nargs="?", help="Actual data Parquet file")
    parser.add_argument(
        "--absolute-tolerance",
        type=float,
        default=0.1,
        help="Allowed absolute position difference (default: 0.1)",
    )
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=0.0,
        help="Allowed relative position difference (default: 0.0)",
    )
    args = parser.parse_args()
    path_file = args.path_file or input("Planned path Parquet file: ").strip()
    actual_file = args.actual_file or input("Actual drone data Parquet file: ").strip()
    compare_paths(
        path_file,
        actual_file,
        absolute_tolerance=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
    )


if __name__ == "__main__":
    main()
