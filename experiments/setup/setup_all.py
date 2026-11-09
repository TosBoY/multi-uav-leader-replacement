"""Regenerate all experiment registry tables."""

try:
    from experiments.setup.feasible import make_feasible_table
    from experiments.setup.formations import make_formations_table
    from experiments.setup.init_database import initialize_database
    from experiments.setup.paths import make_paths_table
    from experiments.setup.runs import make_runs_table
    from experiments.setup.runs_result import make_runs_result_table
except ModuleNotFoundError:
    from feasible import make_feasible_table
    from formations import make_formations_table
    from init_database import initialize_database
    from paths import make_paths_table
    from runs import make_runs_table
    from runs_result import make_runs_result_table


def setup_all() -> None:
    """Regenerate setup registries in dependency order."""
    print(f"Created {initialize_database()}")
    print(f"Created registry.paths in {make_paths_table()}")
    print(f"Created registry.formations in {make_formations_table()}")
    print(f"Created registry.runs in {make_runs_table()}")
    print(f"Created registry.runs_result in {make_runs_result_table()}")
    print(f"Created registry.feasible in {make_feasible_table()}")


if __name__ == "__main__":
    setup_all()
