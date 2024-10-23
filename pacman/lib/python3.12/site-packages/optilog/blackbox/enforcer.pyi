import abc
from _typeshed import Incomplete as Incomplete
from abc import ABC, abstractmethod
from pathlib import Path

class ExecutionConstraints:
    """Describes the execution constraints of a process
    """
    h_wall_time: Incomplete
    h_cpu_time: Incomplete
    h_virtual_memory: Incomplete
    h_real_memory: Incomplete
    s_wall_time: Incomplete
    s_cpu_time: Incomplete
    s_virtual_memory: Incomplete
    s_real_memory: Incomplete
    enforcer: Incomplete
    def __init__(self, enforcer: Enforcer, h_wall_time: int = None, h_cpu_time: int = None, h_virtual_memory: str = None, h_real_memory: str = None, s_wall_time: int = None, s_cpu_time: int = None, s_virtual_memory: str = None, s_real_memory: str = None) -> None:
        """
        :param h_wall_time: Hard Wall time limit in seconds
        :param s_wall_time: Soft Wall time limit in seconds
        :param h_cpu_time: Hard CPU time limit in seconds
        :param s_cpu_time: Soft CPU time limit in seconds
        :param h_virtual_memory: Hard virtual memory limit in seconds
        :param s_virtual_memory: Soft virtual memory limit in seconds
        :param h_real_memory: Hard real memory limit (rss+swap) in seconds
        :param s_real_memory: Soft real memory limit (rss+swap) in seconds
        :param enforcer: Enforcer object that ensures the execution constraints are respected.
        """

class Enforcer(ABC, metaclass=abc.ABCMeta):
    """
    Abstract class that enfoces the execution constraints on a given command
    """
    @abstractmethod
    def assert_support(self, constraints: ExecutionConstraints): ...
    @abstractmethod
    def format(self, args: list[str], constraints: ExecutionConstraints): ...
    @abstractmethod
    def parse_exit(self, exit_code: int): ...
    @abstractmethod
    def get_watcher_file(self): ...
    @abstractmethod
    def delete_watcher_file(self): ...

class RunSolver(Enforcer):
    """
    RunSolver class that implements the Enforcer interface and ensures the execution constraints are satisfied.
    
    .. warning::
        Note that RunSolver is a separate program not included with OptiLog.
        You can download it from http://www.cril.univ-artois.fr/~roussel/runsolver/

    RunSolver actively monitors the full tree of processes spawned by the command and kills them if the constraints are not respected.
    In order to do so, the process will wake up routinely and check the constraints.

    This may cause a slight overhead in cache misses and context switches. A way to reduce this overhead is to reseve an extra core for the runsolver process.

    You can only set the soft constraints on RunSolver because the hard constraints are enforced after ``delay`` seconds are passed.
    """
    path: Incomplete
    delay: Incomplete
    signal: Incomplete
    user_watcher_path: Incomplete
    tmp_watcher: Incomplete
    watcher_path: Incomplete
    timestamp: Incomplete
    def __init__(self, path: Path = None, delay: int = None, watcher_path: str | Path = None, timestamp: bool = False) -> None:
        """
        :param path: Path to the runsolver executable. If None, the path will be searched in the PATH environment variable.
        :param delay: Delay in seconds between checking the constraints. If None, the default value of runsolver will be used.
        :param watcher_path: Path to the watcher file. If None, a temporary file will be created.
        :param timestamp: If True, the data will have the wall time and cpu time attached to it, also stdout and stderr will be merged into stdout.
        """
    def assert_support(self, constraints: ExecutionConstraints): ...
    def format(self, args: list[str], constraints: ExecutionConstraints): ...
    def parse_exit(self, exit_code: int): ...
    def get_watcher_file(self) -> None: ...
    def delete_watcher_file(self) -> None: ...

class DockerEnforcer(Enforcer):
    """
    DockerEnforcer class that implements the Enforcer interface and ensures the execution constraints are satisfied.
    This class uses Docker to enforce the execution constraints.
    """
    def assert_support(self, constraints: ExecutionConstraints): ...
    def format(self, args: list[str], constraints: ExecutionConstraints): ...
    def parse_exit(self, exit_code: int): ...
    def get_watcher_file(self) -> None: ...
    def delete_watcher_file(self) -> None: ...

class ULimit(Enforcer, metaclass=abc.ABCMeta): ...
