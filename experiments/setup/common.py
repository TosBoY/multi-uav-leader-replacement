"""Shared helpers for experiment setup tables."""

from collections.abc import Iterable, Mapping, Sequence
import csv
from pathlib import Path
from typing import Any

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "results" / "experiments.duckdb"
REGISTRY_SCHEMA = "registry"


def connect_database() -> duckdb.DuckDBPyConnection:
    """Open the persistent experiment database and ensure its schemas exist."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(DATABASE_PATH))
    connection.execute("CREATE SCHEMA IF NOT EXISTS registry")
    connection.execute("CREATE SCHEMA IF NOT EXISTS analysis")
    return connection


def read_csv_rows(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV file and return its columns and rows."""
    source = Path(path)
    with source.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = list(reader.fieldnames or [])
        return fieldnames, list(reader)


def write_csv_rows(
    path: str | Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> Path:
    """Write rows to a CSV file, creating its parent directory if needed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
    return destination


def validate_columns(
    fieldnames: Sequence[str],
    expected: Sequence[str],
) -> None:
    """Raise an error when a table or CSV has the wrong columns."""
    if list(fieldnames) != list(expected):
        raise ValueError(
            f"Columns must be {list(expected)}, got {list(fieldnames)}."
        )


def validate_choice(value: str, allowed: Sequence[str], field_name: str) -> str:
    """Validate and return a value from a fixed set of choices."""
    if value not in allowed:
        choices = ", ".join(repr(choice) for choice in allowed)
        raise ValueError(f"{field_name} must be one of: {choices}.")
    return value
