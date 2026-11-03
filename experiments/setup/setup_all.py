"""Regenerate all experiment registry CSV files."""

import time

try:
    from experiments.setup.feasible import make_feasible_csv
    from experiments.setup.formations import make_formations_csv
    from experiments.setup.paths import make_paths_csv
    from experiments.setup.runs import make_runs_csv
    from experiments.setup.runs_result import make_runs_result_csv
except ModuleNotFoundError:
    from feasible import make_feasible_csv
    from formations import make_formations_csv
    from paths import make_paths_csv
    from runs import make_runs_csv
    from runs_result import make_runs_result_csv


def setup_all() -> None:
    """Regenerate setup registries in dependency order."""
    print(f"Created {make_paths_csv()}")
    print(f"Created {make_formations_csv()}")
    print("Waiting before creating runs.csv...")
    time.sleep(3.0)
    print(f"Created {make_runs_csv()}")
    print(f"Created {make_runs_result_csv()}")
    print(f"Created {make_feasible_csv()}")


if __name__ == "__main__":
    setup_all()
