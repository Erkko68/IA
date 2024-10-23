import abc
from _typeshed import Incomplete as Incomplete
from abc import ABC, abstractmethod
from typing import Callable, NamedTuple

NAMESPACE_SEPARATOR: str

class Condition(NamedTuple):
    child: Incomplete
    param: Incomplete
    value: Incomplete

RESERVERD_WORDS: Incomplete

class Parameter(ABC, metaclass=abc.ABCMeta):
    def __init__(self) -> None: ...
    @property
    @abc.abstractmethod
    def has_default(self): ...
    def get_params(self) -> None: ...
    @abstractmethod
    def get_params_dict(self): ...
    def get_constraints(self) -> None: ...
    @abstractmethod
    def in_domain(self, item): ...
    @abstractmethod
    def instantiate(self, cfg): ...
    @abstractmethod
    def __eq__(self, o): ...
    default: Incomplete
    def inject(self, default) -> None: ...

class Int(Parameter):
    """Annotates an integer parameter.
    """
    has_default: bool
    start: Incomplete
    end: Incomplete
    default: Incomplete
    def __init__(self, start: int, end: int, default: int = None) -> None:
        """
        :param start: Start of parameter range (inclusive)
        :param end: End of parameter range (inclusive)
        :param default: Default value of the parameter.             **Note:** May be specified in the definition of the function.

        .. code-block:: python

            >>> from optilog.tuning import ac, Int
            >>> @ac
            >>> def func(data, seed, p: Int(-10, 10) = 0):
            >>>     ...
        """
    def in_domain(self, item) -> None: ...
    def instantiate(self, item) -> None: ...
    def get_params_dict(self) -> None: ...
    def __eq__(self, o): ...
    @staticmethod
    def from_params_dict(params) -> None: ...

class Categorical(Parameter):
    """Annotates a categorical parameter.
    """
    has_default: bool
    options: Incomplete
    default: Incomplete
    def __init__(self, *args: list[str], default: str = None) -> None:
        '''
        :param args: List of possible categorical values
        :param default: Default value of the parameter.             **Note:** May be specified in the definition of the function.

        .. code-block:: python

            >>> from optilog.tuning import ac, Categorical
            >>> @ac
            >>> def func(data, seed, p: Categorical("A", "B", "C") = "A"):
            >>>     ...
        '''
    def in_domain(self, item) -> None: ...
    def instantiate(self, item) -> None: ...
    def get_params_dict(self) -> None: ...
    def __eq__(self, o): ...
    @staticmethod
    def from_params_dict(params) -> None: ...

class Dict(Parameter):
    """
    Annotates a set of parameters all at once
    """
    has_default: bool
    dt: Incomplete
    default: Incomplete
    def __init__(self, dt: dict[str, Parameter]) -> None:
        """
        :param dt: Dictionary of pairs (string, parameter) to configure.

        .. code-block:: python

            >>> @ac
            >>> def func(data, seed,
            >>>          d: Dict({'height': Int(3,10, default=7), 'width': Int(1, 8, default=6)})):
            >>>     ...
        """
    def in_domain(self, item) -> None: ...
    def instantiate(self, cfg) -> None: ...
    def __eq__(self, o): ...
    def get_params_dict(self) -> None: ...
    def get_constraints(self) -> None: ...

class Choice(Parameter):
    """
    Annotates a disjunctive parameter (enum-union like, or also known as algebraic data types)
    """
    has_default: bool
    choices: Incomplete
    categorical: Incomplete
    default: Incomplete
    def __init__(self, *args: list[Parameter], default: Parameter = None) -> None:
        """
        :param args: Parameters to configure.
        :param default: Default value of the parameter. Defaults to None.             **Note:** May be specified in the definition of the function.

        .. code-block:: python

            >>> from optilog.tuning import ac, Int, Real, Choice
            >>>
            >>> @ac
            >>> def func(
            >>>     data, seed,
            >>>     n: Choice(Int(-7, 8, default=3), Real(0.1, 4.8, default=1.0)) = Int(-7, 8, default=3),
            >>> ):
            >>>     ...
        """
    @staticmethod
    def from_params_dict(params) -> None: ...
    def in_domain(self, item) -> None: ...
    def instantiate(self, cfg) -> None: ...
    def __eq__(self, o): ...
    def get_params_dict(self) -> None: ...
    def get_constraints(self) -> None: ...

class NamedChoice(Parameter):
    """
    Annotates a disjunctive parameter (enum-union like, or also known as algebraic data types)
    Each parameter is associated with a name, which allows to distinguish which disjunction is selected
    """
    has_default: bool
    choices: Incomplete
    categorical: Incomplete
    default: Incomplete
    def __init__(self, choices: dict[str, Parameter], default: str = None) -> None:
        '''
        :param args: Parameters to configure prepended by the name of the choice.
        :param default: Default value of the parameter. Defaults to None.             **Note:** May be specified in the definition of the function.

        .. code-block:: python

            >>> from optilog.tuning import ac, Int, Real, Choice
            >>>
            >>> @ac
            >>> def func(
            >>>     data, seed,
            >>>     n: NamedChoice({"v1": Int(-7, 8, default=3), "v2": Real(0.1, 4.8, default=1.0)}) = "v1",
            >>> ):
            >>>     ...
        '''
    @staticmethod
    def from_params_dict(params) -> None: ...
    def in_domain(self, item) -> None: ...
    def instantiate(self, cfg) -> None: ...
    def __eq__(self, o): ...
    def get_params_dict(self) -> None: ...
    def get_constraints(self) -> None: ...

class Bool(Parameter):
    """
    Annotates a boolean parameter.
    """
    has_default: bool
    default: Incomplete
    def __init__(self, default: bool = None) -> None:
        """
        :param default: Default value of the parameter. Defaults to None.             **Note:** May be specified in the definition of the function.

        .. code-block:: python

            >>> from optilog.tuning import ac, Bool
            >>> @ac
            >>> def func(data, seed, p: Bool() = True):
            >>>     ...
        """
    def in_domain(self, item) -> None: ...
    def instantiate(self, item) -> None: ...
    def get_params_dict(self) -> None: ...
    def __eq__(self, o): ...
    @staticmethod
    def from_params_dict(params) -> None: ...

class Real(Parameter):
    """Annotates a real parameter.
    """
    has_default: bool
    start: Incomplete
    end: Incomplete
    default: Incomplete
    def __init__(self, start: float, end: float, default: float = None) -> None:
        """
        :param start: Start of parameter range (inclusive)
        :param end: End of parameter range (inclusive)
        :param default: Default value of the parameter.             **Note:** May be specified in the definition of the function.

        .. code-block:: python

            >>> from optilog.tuning import ac, Real
            >>> @ac
            >>> def func(data, seed, p: Real(-1.3, 3.14) = 0.18):
            >>>     ...
        """
    def in_domain(self, item) -> None: ...
    def instantiate(self, item) -> None: ...
    def get_params_dict(self) -> None: ...
    def __eq__(self, o): ...
    @staticmethod
    def from_params_dict(params) -> None: ...

class CfgCall(Parameter):
    """
    Configures a local configurable function.
    See the documentation for `Local Configurable Functions`
    """
    has_default: bool
    fn: Incomplete
    default: Incomplete
    dt_params: Incomplete
    keyword: Incomplete
    def __init__(self, fn: Callable) -> None:
        """
        :param fn: Function to configure

        .. code-block:: python

            >>> from optilog.tuning import ac, CfgCall
            >>> @ac
            >>> def func2(
            >>>     data, seed,
            >>>     func1_call1: CfgCall(func1),
            >>>     func1_call2: CfgCall(func1),
            >>> ):
        """
    def in_domain(self, item) -> None: ...
    def get_constraints(self) -> None: ...
    def __eq__(self, o): ...
    def instantiate(self, cfg) -> None: ...
    def get_params_dict(self) -> None: ...
    def __call__(self, *args, **kwargs) -> None: ...

class CfgCls(Parameter):
    """Annotates a configurable class
    """
    has_default: bool
    cls: Incomplete
    default: Incomplete
    dt_params: Incomplete
    def __init__(self, cls: type) -> None:
        """
        :param cls: Configurable class

        .. code-block:: python

            >>> from optilog.solvers.sat import Glucose41
            >>> @ac
            >>> def func(
            >>>     instance,
            >>>     seed,
            >>>     init_solver_fn: CfgCls(Glucose41),
            >>> ):
            >>>     solver = init_solver_fn(seed=...)
            >>>     ...
        """
    def in_domain(self, item) -> None: ...
    def get_constraints(self) -> None: ...
    def __eq__(self, o): ...
    def instantiate(self, cfg) -> None: ...
    def get_params_dict(self) -> None: ...
    def __call__(self, *args, **kwargs) -> None: ...

class CfgObj(CfgCls):
    """Annotates a configurable class and instanties it
    """
    def __init__(self, cls: type) -> None:
        """
        :param cls: Configurable class

        .. code-block:: python

            >>> from optilog.solvers.sat import Glucose41
            >>> @ac
            >>> def func(
            >>>     instance,
            >>>     seed,
            >>>     solver: CfgObj(Glucose41),
            >>> ):
            >>>     ...
        """
    def __eq__(self, o): ...
    def instantiate(self, cfg) -> None: ...
    def __call__(self, *args, **kwargs) -> None: ...

class GlobalConfigurableFn:
    fn: Incomplete
    def __init__(self, fn, constraints) -> None: ...
    def configure_default(self, cfg) -> None: ...
    def get_params(self) -> None: ...
    def get_original_fn(self) -> None: ...

def get_params_obj_fn(fn) -> None: ...
def get_params_target(o) -> None: ...
def get_constraints_target(o) -> None: ...
def get_fn_provided_args(fn, args) -> None: ...
def call_with_injectable(fn, injectable, keyword, *args, **kwargs) -> None: ...
def ac(*args, constraints: Incomplete | None = None): ...
def merge_dict_rec(a, b) -> None: ...
def constraints_flatten(o, namespace) -> None: ...
def config_flatten(o) -> None: ...
def config_str_to_dict(config) -> None: ...
def get_params_as_obj(d) -> None: ...
