"""Initialize the persistent DuckDB database used by experiments."""

try:
    from experiments.setup.common import DATABASE_PATH, connect_database
except ModuleNotFoundError:
    from common import DATABASE_PATH, connect_database


def initialize_database():
    """Create ``results/experiments.duckdb`` and its application schemas."""
    with connect_database():
        pass
    return DATABASE_PATH


if __name__ == "__main__":
    print(f"Initialized {initialize_database()}")
