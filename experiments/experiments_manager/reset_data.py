"""Delete generated simulation data from results/data."""

from pathlib import Path

try:
    from experiments.setup.setup_all import setup_all
except ModuleNotFoundError:
    from pathlib import Path as _Path
    import sys

    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    from experiments.setup.setup_all import setup_all


_RESULTS_DATA_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "results" / "data"
)


def delete_results_data() -> None:
    """Delete every file inside results/data while preserving directories."""
    _RESULTS_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for child in _RESULTS_DATA_DIRECTORY.rglob("*"):
        if child.is_file() or child.is_symlink():
            child.unlink()


if __name__ == "__main__":
    delete_results_data()
    print(f"Deleted all data from {_RESULTS_DATA_DIRECTORY}")
    setup_all()
