"""Shared helpers for experiment setup tables."""

from collections.abc import Iterable, Mapping, Sequence
import csv
from pathlib import Path
from typing import Any


REGISTRY_DIRECTORY = Path(__file__).resolve().parents[2] / "results" / "registry"


def setup_path(filename: str) -> Path:
    """Return a path inside the project experiment registry directory."""
    return REGISTRY_DIRECTORY / filename


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
    """Raise an error when a setup CSV has the wrong columns."""
    if list(fieldnames) != list(expected):
        raise ValueError(
            f"CSV columns must be {list(expected)}, got {list(fieldnames)}."
        )


def validate_choice(value: str, allowed: Sequence[str], field_name: str) -> str:
    """Validate and return a value from a fixed set of choices."""
    if value not in allowed:
        choices = ", ".join(repr(choice) for choice in allowed)
        raise ValueError(f"{field_name} must be one of: {choices}.")
    return value
