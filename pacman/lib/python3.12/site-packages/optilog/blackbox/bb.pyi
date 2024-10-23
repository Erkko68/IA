import abc
from .enforcer import ExecutionConstraints as ExecutionConstraints
from _typeshed import Incomplete as Incomplete
from abc import ABC, abstractmethod
from enum import Enum
from optilog.formulas import CNF as CNF
from typing import NamedTuple

class BlackBoxRedirection(Enum):
    """
    Blackbox redirection options
    """
    Default = 1
    Stdout = 2
    Null = 3
    Process = 4

class BlackBoxArgument(Enum):
    Instance = '__BLACKBOX_SUBS_INSTANCE__'
    Seed = '__BLACKBOX_SUBS_SEED__'
    Memory = '__BLACKBOX_SUBS_MEMORY__'
    WallTime = '__BLACKBOX_SUBS_WALLTIME__'
    CPUTime = '__BLACKBOX_SUBS_CPUTIME__'

class ClassOrInstanceMethod:
    wrapped: Incomplete
    def __init__(self, wrapped) -> None: ...
    def __get__(self, instance, owner) -> None: ...

class BlackBox(ABC):
    configured: Incomplete
    def __init__(self) -> None: ...
    def get_config(cls) -> dict[str, dict[str, str]]:
        """
        :return: A dictionary with all the configurable parameters of the solver.
            Each parameter has defined its type (*int*, *float*, *bool* or *categorical*), its domain, and its default value.
        """
    def set_default(cls, key: str, value, check_domain: bool = True):
        """
        Sets the value for a given parameter in the default configuration

        :param key: Parameter name.
        :param value: Parameter value.
        :param value: int or float or bool or str
        :raises: KeyError: if ``param`` is not found inside the blackbox.
        """
    def configure(self, cfg) -> None: ...
    def configure_default(cls, cfg) -> None: ...
    def get_constraints(cls) -> None: ...
    def get(self, key: str) -> int | float | bool | str:
        """Returns the value assigned to a given parameter name.

        :param key: Parameter name.
        :return: Parameter value.
        """
    def set(self, key: str, value, check_domain: bool = True):
        """
        Sets the value for a given parameter.

        :param key: Parameter name.
        :param value: Parameter value.
        :param value: int or float or bool or str
        :raises: KeyError: if ``param`` is not found inside the blackbox.
        """
    def run(self) -> None: ...

class PythonBlackBox(BlackBox):
    fn: Incomplete
    def __init__(self, fn) -> None: ...
    def get_config(self) -> dict[str, dict[str, str]]: ...
    def get_constraints(self) -> None: ...
    def run(self, *args, **kwargs) -> None: ...

class ParsingBlackBox(BlackBox, metaclass=abc.ABCMeta):
    def __init__(self) -> None: ...
    @abstractmethod
    def get_parsing_info(self): ...
    @property
    @abc.abstractmethod
    def stdout(self): ...
    @property
    @abc.abstractmethod
    def stderr(self): ...
    def reset_parsing(self) -> None: ...
    def parse_output(self) -> None: ...
    def __getattr__(self, key) -> None: ...
    def run(self) -> None: ...
    @property
    def parsed(self) -> None:
        """
        Dictionary with all the parsed elements from the execution.
        Accessible after the execution has finished.
        """

class DisableConstraints: ...

class SystemBlackBox(ParsingBlackBox):
    """
    Blackbox that runs an external subprocess (usually a binary).
    """
    Instance: Incomplete
    Seed: Incomplete
    WallTime: Incomplete
    CPUTime: Incomplete
    Memory: Incomplete
    SUBS_ARGS: Incomplete
    arguments: Incomplete
    constraints: Incomplete
    completed: Incomplete
    completed_enforcer: Incomplete
    parsing_info: Incomplete
    def __init__(self, arguments: list[str | BlackBoxArgument], unbuffer: str | bool = False, constraints: ExecutionConstraints | None = None, parsing_info: ParsingInfo | None = None) -> None:
        """
        :param arguments: List of arguments for the command to be executed. Placeholders might be used, such as
            `SystemBlackBox.Instance` and `SystemBlackBox.Seed`, that will be replaced during execution.
        
        :param unbuffer: Path to the unbuffer binary, which may be desirable to unbuffer the pipe
            of the call. If it is a boolean, it will search `unbuffer` in the `PATH`. See the
            `unbuffer documentation <https://www.tcl.tk/man/expect5.31/unbuffer.1.html/>`_
            for more information.
        :param constraints: |constraints-param|
        :param parsing_info: Optional ParsingInfo object used for parsing the output of the blackbox
        """
    @property
    def unbuffer(self) -> None: ...
    @unbuffer.setter
    def unbuffer(self, value) -> None: ...
    def get_parsing_info(self) -> None: ...
    @property
    def returncode(self) -> int:
        """Return code of the execution
        Accessible after the execution has finished.
        """
    @property
    def stdout(self) -> str:
        """Stdout output of the execution.
        Accessible after the execution has finished.
        """
    @property
    def stderr(self) -> str:
        """Stderr output of the execution.
        Accessible after the execution has finished.
        """
    def format_config(self, args: list[str]):
        """
        Formats the configurable parameters of the blackbox, and concats them with `args`.
        By default, all parameters are formatted as ``--param-name=value``.
        This method is expected to be overwritten for commands that expect the parameters in a different format.

        :param list args: List of current arguments
        :return: Final list of strings with the parameters to be called
        """
    def run(self, instance: str, seed: int = None, constraints: ExecutionConstraints = None, stdout: str | BlackBoxRedirection = ..., stderr: str | BlackBoxRedirection = ..., text: bool = True):
        """Executes the blackbox and stores its standard output

        :param instance: Instance to execute
        :param seed: Seed of the execution
        :param constraints: |constraints-param|. If it is None, the default constraints of the constructor will be used. Otherwise, these new constraints will be used.
        :param stdout: Where to redirect stdout. If it is a string, the output will be redirected to the specified path.
        :param stderr: Where to redirect stderr. If it is a string, the output will be redirected to the specified path.
        :param text: Whether the output is text based or binary
        """

class SatexArgsInfo(NamedTuple):
    pattern: Incomplete
    refresh_list: Incomplete

class SatexArgsInfoNoPattern(NamedTuple):
    all: Incomplete
    refresh_list: Incomplete

class SatexArgsRun(NamedTuple):
    pretend: Incomplete
    pull: Incomplete

class SatexArgsRunTimeout(NamedTuple):
    pretend: Incomplete
    pull: Incomplete
    timeout: Incomplete
    fail_if_timeout: Incomplete

class SatexBlackBox(ParsingBlackBox):
    solver: Incomplete
    repo: Incomplete
    constraints: Incomplete
    completed: Incomplete
    def __init__(self, solver: str, constraints: ExecutionConstraints = None) -> None:
        """
        :param solver: Name of the solver to insantiate. See the method ``available_solvers``
        :param constraints: |constraints-param|
        """
    @staticmethod
    def available_solvers(pattern: str | None = None) -> list[str]:
        """
        SatexBlackBox accesses the SAT solvers used by `satex <https://github.com/sat-heritage/docker-images>`_
        This method returns a list of available solvers that match the given pattern.

        :param pattern: Pattern for available solvers
        :return: List of strings with the names of available solvers

        .. code-block:: python

            >>> from optilog.blackbox import SatexBlackBox
            >>>
            >>> SatexBlackBox.available_solvers('kissat*')
            ['kissat-sc2020-default:2020', 'kissat-sc2020-sat:2020', 'kissat-sc2020-unsat:2020',
            'kissat-mab:2021', 'kissat-sat_crvr_gb:2021', 'kissat-sc2021:2021', 'kissat_bonus:2021',
            'kissat_cf:2021', 'kissat_gb:2021']
        """
    @property
    def is_sat(self) -> bool:
        """
        Whether the solver has determined the instance is satisfiable or not.
        Accessible after the execution has finished.
        """
    @property
    def sat(self) -> str:
        """
        Result of the status line of the solver.
        Accessible after the execution has finished.
        """
    @property
    def model(self) -> list[int]:
        """
        Parsed model as list of integers of the instance
        Accessible after the execution has finished.
        """
    @property
    def stdout(self) -> str:
        """
        Stdout output of the execution.
        Accessible after the execution has finished.
        """
    @property
    def stderr(self) -> str:
        """
        Stderr output of the execution.
        Accessible after the execution has finished.
        """
    def get_parsing_info(self) -> None: ...
    @property
    def returncode(self) -> None: ...
    @property
    def stdout(self) -> None: ...
    @property
    def stderr(self) -> None: ...
    def run(self, instance: str | CNF, constraints: ExecutionConstraints = None, stdout: str | BlackBoxRedirection = ..., stderr: str | BlackBoxRedirection = ...):
        """
        Executes the SAT solver and stores the output out of the call

        Note that this call will launch a docker container and may pull a docker image. Startup may not be instantaneous.

        :param instance: Path to the CNF instance to execute. It may also be a CNF object, in which case it will
            be written to a memory file without going through disk.
            Note that this might would result in faster loading times but higher memory consumption.
        :param constraints: |constraints-param|. If it is None, the default constraints of the constructor will be used. Otherwise, these new constraints will be used.
        :param stdout: Where to redirect stdout. If it is a string, the output will be redirected to the specified path.
        :param stderr: Where to redirect stderr. If it is a string, the output will be redirected to the specified path.
        """
