"""Generate and edit the experiment runs setup table."""

from collections.abc import Iterable, Iterator, Mapping, Sequence
import csv
from pathlib import Path
import tempfile

try:
    from experiments.setup.common import (
        REGISTRY_DIRECTORY,
        setup_path,
        validate_choice,
        validate_columns,
        write_csv_rows,
    )
except ModuleNotFoundError:
    from common import (
        REGISTRY_DIRECTORY,
        setup_path,
        validate_choice,
        validate_columns,
        write_csv_rows,
    )


RUN_COLUMNS = [
    "runs_id",
    "path_id",
    "formation_id",
    "leader_failure_enabled",
    "fail_time",
    "init_leader",
    "election_type",
    "status",
]
ELECTION_TYPES = ("closest", "minimum_travel")
RUN_STATUSES = ("planned", "running", "done", "failed", "infeasible", "cancelled")
FAIL_TIMES = tuple(2.5 + 0.5 * index for index in range(11))
DEFAULT_PATHS_FILE = setup_path("paths.csv")
DEFAULT_FORMATIONS_FILE = setup_path("formations.csv")
DEFAULT_RUNS_FILE = REGISTRY_DIRECTORY / "runs.csv"


def _read_ids(path: str | Path, id_column: str) -> list[int]:
    """Read one integer identifier column from a registry CSV."""
    with Path(path).open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        validate_columns(list(reader.fieldnames or []), [id_column, *[field for field in reader.fieldnames or [] if field != id_column]])
        values = [int(row[id_column]) for row in reader]
    if not values:
        raise ValueError(f"{path} does not contain any rows.")
    return values


def _run_rows(
    path_ids: Iterable[int],
    formation_ids: Iterable[int],
) -> Iterator[dict[str, object]]:
    """Yield all run rows in path-major, formation-minor order."""
    runs_id = 1
    for path_id in path_ids:
        for formation_id in formation_ids:
            yield {
                "runs_id": runs_id,
                "path_id": path_id,
                "formation_id": formation_id,
                "leader_failure_enabled": False,
                "fail_time": "",
                "init_leader": 0,
                "election_type": ELECTION_TYPES[0],
                "status": "planned",
            }
            runs_id += 1
            for fail_time in FAIL_TIMES:
                for election_type in ELECTION_TYPES:
                    yield {
                        "runs_id": runs_id,
                        "path_id": path_id,
                        "formation_id": formation_id,
                        "leader_failure_enabled": True,
                        "fail_time": fail_time,
                        "init_leader": 0,
                        "election_type": election_type,
                        "status": "planned",
                    }
                    runs_id += 1


def generate_runs(
    paths_path: str | Path = DEFAULT_PATHS_FILE,
    formations_path: str | Path = DEFAULT_FORMATIONS_FILE,
) -> Iterator[dict[str, object]]:
    """Yield runs for every path/formation pair."""
    path_ids = _read_ids(paths_path, "path_id")
    formation_ids = _read_ids(formations_path, "formation_id")
    return _run_rows(path_ids, formation_ids)


def make_runs_csv(
    output_path: str | Path = DEFAULT_RUNS_FILE,
    paths_path: str | Path = DEFAULT_PATHS_FILE,
    formations_path: str | Path = DEFAULT_FORMATIONS_FILE,
) -> Path:
    """Stream all generated runs into the runs registry CSV."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RUN_COLUMNS)
        writer.writeheader()
        writer.writerows(generate_runs(paths_path, formations_path))
    return destination


def edit_run(
    runs_id: int,
    output_path: str | Path = DEFAULT_RUNS_FILE,
    **updates: object,
) -> Path:
    """Edit one run row while streaming the remaining rows unchanged."""
    if not isinstance(runs_id, int) or isinstance(runs_id, bool):
        raise TypeError("runs_id must be an integer.")
    if any(field not in RUN_COLUMNS[1:] for field in updates):
        raise ValueError("Updates contain an unknown or non-editable field.")
    if "election_type" in updates:
        validate_choice(str(updates["election_type"]), ELECTION_TYPES, "election_type")
    if "status" in updates:
        validate_choice(str(updates["status"]), RUN_STATUSES, "status")

    source = Path(output_path)
    if not source.exists():
        raise FileNotFoundError(source)

    changed = False
    with tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        dir=source.parent,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        writer = csv.DictWriter(temporary_file, fieldnames=RUN_COLUMNS)
        with source.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            validate_columns(list(reader.fieldnames or []), RUN_COLUMNS)
            writer.writeheader()
            for row in reader:
                if int(row["runs_id"]) == runs_id:
                    row.update({field: str(value) for field, value in updates.items()})
                    changed = True
                writer.writerow(row)

    if not changed:
        temporary_path.unlink(missing_ok=True)
        raise ValueError(f"No run with runs_id={runs_id} exists.")
    temporary_path.replace(source)
    return source


def edit_run_status(
    run_id: int,
    new_status: str,
    output_path: str | Path = DEFAULT_RUNS_FILE,
) -> Path:
    """Update the status of one run for the experiment runner."""
    if not isinstance(run_id, int) or isinstance(run_id, bool):
        raise TypeError("run_id must be an integer.")
    validate_choice(new_status, RUN_STATUSES, "new_status")
    return edit_run(run_id, output_path, status=new_status)


def ensure_status_column(output_path: str | Path = DEFAULT_RUNS_FILE) -> Path:
    """Add ``status=planned`` to an older runs CSV in a streaming pass."""
    source = Path(output_path)
    if not source.exists():
        raise FileNotFoundError(source)

    with source.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = list(reader.fieldnames or [])
        if fieldnames == RUN_COLUMNS:
            return source
        validate_columns(fieldnames, RUN_COLUMNS[:-1])

        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=source.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            writer = csv.DictWriter(temporary_file, fieldnames=RUN_COLUMNS)
            writer.writeheader()
            for row in reader:
                row["status"] = "planned"
                writer.writerow(row)

    temporary_path.replace(source)
    return source


if __name__ == "__main__":
    if DEFAULT_RUNS_FILE.exists():
        ensure_status_column()
    else:
        make_runs_csv()
