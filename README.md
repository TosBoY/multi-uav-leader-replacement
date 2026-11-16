# Multi-UAV Leader Replacement

Comparative evaluation of leader replacement strategies in multi-UAV
formation control using RotorPy.

## Thesis

This project implements and evaluates multiple leader replacement
strategies for a leader-follower UAV formation after leader failure.

## Simulator

RotorPy

## Language

Python 3.11

## Planned Strategies

- Static pre-assigned order
- Proximity to formation center
- Remaining energy
- Communication connectivity
- No-replacement baseline

## Planned Experiments

- Line formation
- V formation
- 5, 8, and 10 UAVs
- Leader failure during straight flight
- Leader failure during maneuver

## Metrics

- Formation error
- Trajectory tracking error
- Recovery time
- Mission success rate

## Reproducing the experiments on another machine

The repository does not include the local `.venv/` environment or generated
experiment output. Create a new environment and regenerate the experiment
database after cloning.

### Windows PowerShell setup

Install Git and Python 3.11 first, then run:

```powershell
git clone https://github.com/TosBoY/multi-uav-leader-replacement.git
cd multi-uav-leader-replacement
code .
```

In VS Code, install the **Python** extension from Microsoft. Pylance is
recommended for IntelliSense and diagnostics. Open a new VS Code terminal
(`Ctrl+Shift+``) and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

After installation, select the project environment with `Ctrl+Shift+P` →
**Python: Select Interpreter** → select:

```text
.venv\Scripts\python.exe
```

If `.venv` does not appear in the interpreter list, reload VS Code and select
**Python: Select Interpreter** again. The repository's VS Code settings already
point to `.venv\Scripts\python.exe`.

### Run with the VS Code Play button

The repository includes VS Code launch configurations. Open the **Run and
Debug** view (`Ctrl+Shift+D`), choose a configuration from the dropdown, and
press the green Play button:

- **Run current Python file** runs the file currently open in the editor.
- **Initialize experiment registry** creates the local experiment database.
- **Run experiment batch** runs the ranges configured in
  `experiments\experiments_manager\experiment_run.py`.
- **Display registered experiment** prompts for a run ID and displays it.

The editor's top-right play button also runs the currently open Python file.
Make sure the selected interpreter is the project `.venv` before running.

If PowerShell blocks activation, either use the environment's interpreter
directly (for example, `.\.venv\Scripts\python.exe`) or allow scripts for the
current user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Initialize the experiment registry

The registry database is generated locally and is required before running
registered experiments:

```powershell
python -m experiments.setup.setup_all
```

This creates `results\experiments.duckdb` and the experiment registry tables.
The command is safe to rerun when starting from a fresh checkout.

### Run an experiment

To run one registered experiment with a display and without saving output:

```powershell
python -m experiments.experiments_manager.display_run
```

Enter the run ID when prompted.

For the normal batch workflow, edit the ranges at the top of
`experiments\experiments_manager\experiment_run.py`, then run:

```powershell
python -m experiments.experiments_manager.experiment_run
```

Available batch filters are:

```python
RUN_ID_RANGE = []
PATH_ID_RANGE = []
NUMBER_OF_DRONE_RANGE = []
SHAPE_RANGE = []
PLANE_RANGE = []
SPREAD_RANGE = []
NONFAIL_ONLY = False
```

Empty filters select all values. Two numeric values define an inclusive range,
one numeric value selects an exact value, and string filters select the listed
values. Set `NONFAIL_ONLY = True` to run only experiments with leader failure
disabled; leave it `False` to run both non-failure and failure experiments.

The batch manager runs baseline experiments before their corresponding leader
failure experiments and writes generated data under `results\data\`.

### Calculate analysis metrics

After the experiments have completed, initialize the analysis tables and
calculate trajectory difficulty with:

```powershell
python -m analysis.setup.setup_analysis
python -m analysis.data_analysis.trajectory_difficulty_calculate
```

Trajectory difficulty is calculated from completed non-failure runs using the
leader trajectory (`drone = 0`) and stores three rows per `path_id` and
`fail_time`: one second before failure, one second after failure, and three
seconds after failure. The analysis setup command recreates the analysis
tables, so rerunning it removes previously calculated analysis rows.

### Reset generated data

To remove generated Parquet data and recreate the registry:

```powershell
python -m experiments.experiments_manager.reset_data
```

The `results\` directory is ignored by Git, so outputs remain local to each
machine and are not included in future clones.