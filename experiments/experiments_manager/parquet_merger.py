"""Merge experiment Parquet files and register the combined data file."""

from collections.abc import Sequence
from pathlib import Path

try:
    from experiments.setup.common import PROJECT_ROOT, connect_database
except ModuleNotFoundError:
    from pathlib import Path as _Path
    import sys

    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    from experiments.setup.common import PROJECT_ROOT, connect_database


def merge_parquet_files(folder_path: str | Path, run_ids: Sequence[int]) -> Path:
    """Merge selected run files and add a ``run_id`` column to every row."""
    ids = _validated_run_ids(run_ids)
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(folder)

    is_trajectory_folder = folder.name.lower() == "trajectories"
    suffix = "_path.parquet" if is_trajectory_folder else ".parquet"
    source_paths = [folder / f"{run_id}{suffix}" for run_id in ids]
    missing = [path for path in source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing Parquet files: " + ", ".join(str(path) for path in missing)
        )

    tail = "_path" if is_trajectory_folder else ""
    output_path = folder / f"{ids[0]}_{ids[-1]}{tail}.parquet"
    query_parts: list[str] = []
    parameters: list[object] = []
    for run_id, source_path in zip(ids, source_paths):
        query_parts.append(
            "SELECT *, CAST(? AS BIGINT) AS run_id FROM read_parquet(?)"
        )
        parameters.extend([run_id, str(source_path)])

    with connect_database() as connection:
        connection.execute(
            f"COPY ({' UNION ALL '.join(query_parts)}) "
            "TO ? (FORMAT PARQUET)",
            [*parameters, str(output_path)],
        )
        if folder.name.lower() in {"fail", "nonfail"}:
            placeholders = ", ".join("?" for _ in ids)
            connection.execute(
                f"""
                UPDATE registry.runs_result
                SET data_file = ?
                WHERE run_id IN ({placeholders})
                """,
                [_project_relative_path(output_path), *ids],
            )
    for source_path in source_paths:
        source_path.unlink()
    return output_path


def _validated_run_ids(run_ids: Sequence[int]) -> list[int]:
    """Validate and sort a non-empty collection of unique run IDs."""
    ids = list(run_ids)
    if not ids:
        raise ValueError("run_ids must not be empty.")
    if any(isinstance(run_id, bool) or not isinstance(run_id, int) for run_id in ids):
        raise TypeError("run_ids must contain only integers.")
    if any(run_id < 1 for run_id in ids):
        raise ValueError("run_ids must contain positive integers.")
    if len(set(ids)) != len(ids):
        raise ValueError("run_ids must not contain duplicates.")
    return sorted(ids)


def _project_relative_path(path: Path) -> str:
    """Return a stable project-relative path for registry storage."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    raise SystemExit(
        "Import merge_parquet_files(folder_path, run_ids) from Python instead."
    )