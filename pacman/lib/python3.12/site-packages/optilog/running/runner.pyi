from _typeshed import Incomplete as Incomplete
from optilog.abstractscenario import AbstractScenario as AbstractScenario
from optilog.blackbox import BlackBox as BlackBox, ExecutionConstraints as ExecutionConstraints
from pathlib import Path
from typing import Callable

def contains_common_postfix(test, target) -> None: ...

class CliEntrypoint:
    def __call__(self, scenario) -> None: ...
    @staticmethod
    def execute(scenario, task, seed, solver) -> None: ...
    @staticmethod
    def execute_raw(scenario: RunningScenario, task, seed, solver): ...
    @staticmethod
    def launch(scenario, tasks, seeds, solvers, overwrite) -> None: ...
    @staticmethod
    def clean(scenario, tasks, seeds, solvers) -> None: ...
    @staticmethod
    def list_tasks(scenario) -> None: ...
    @staticmethod
    def list_solvers(scenario) -> None: ...
    @staticmethod
    def list_seeds(scenario) -> None: ...

def running_cli_entrypoint() -> None: ...

class _RunEntrypoint:
    @classmethod
    def run(cls, runner: RunningScenario, task_path: Path, seed: int, solver: str): ...

class RunningSolverType:
    """
    **NOTE: This is not a class, it is a Type Alias**

    This alias can be:
        - A string, representing a path to a binary solver
        - A Python function
        - A PythonBlackBox (object or class)
        - A SystemBlackBox (object or class)
    """
    TYPES = str | BlackBox | Callable | type[BlackBox]

class RunningScenario(AbstractScenario):
    SETTINGS_FILENAME: str
    solvers: Incomplete
    tasks: Incomplete
    seeds: Incomplete
    submit_file: Incomplete
    slots: Incomplete
    working_dir: Incomplete
    timestamp: Incomplete
    unbuffer: Incomplete
    def __init__(self, tasks: str | list[str], submit_file: str, constraints: ExecutionConstraints, solvers: str | dict[str, RunningSolverType] | list[RunningSolverType], logs: str = None, slots: int = 1, seeds: int | list[int] | None = 1, working_dir: str = None, timestamp: bool = True, unbuffer: bool = True) -> None:
        """Handles the creation of the execution scenario.

        Note that all globs in this class are relative to the current working directory.

        :param tasks: Glob string that matches the instances to execute. A list of instances may also be provided.
        :param submit_file: Script used to submit a job to the cluster. The Execution Scenario is agnostic to the system where jobs will be executed. See the :ref:`submit-command-examples` section for some examples of submission commands for different systems
        :param constraints: |constraints-param|
        :param solvers: Either a string, a dictionary or a list.

                        If it is a string, it must represent the glob of a set of solvers.

                        If it is a dictionary, each key must be the name of the solver and each value must be a valid `RunningSolverType`.

                        If it is a list, each value must be a valid `RunningSolverType`.
        :param logs: Path used to save the logs of the execution (both stdout and stderr). By default, they are saved in a `logs` folder in the scenario directory.
        :param slots: Number of slots to reserve on the cluster. Usually corresponds with the number of execution threads.
        :param seeds: List of seeds for the execution.
        :param working_dir: Working directory of execution environment. Defaults to the current working directory.
        :param timestamp: Whether to record the timestamp of every line or not. Possible values are: `False`, for no timestamp; `True` for automatic timestamp; `optilog` for timestamp in OptiLog format or `runsolver` for timestamp in runsolver format. If `True`, it will use RunSolver if it is the current enfocer, and OptiLog otherwise. Note that this will automatically add timestamp as a flag to runsolver.
        :param unbuffer: Whether to force the solver through the `unbuffer` command. `unbuffer` must be in the PATH.
        """
    @property
    def logs(self) -> None: ...
    def list_seeds(self) -> None: ...
    def list_tasks(self) -> None: ...
    def list_solvers(self) -> None: ...
    def generate_scenario(self, scenario_dir: str, log: bool = True):
        """Generates all the files required for the scenario

        :param scenario_dir: Path where the execution scenario will be saved
        :param log: If True, will print a log to the console
        """
