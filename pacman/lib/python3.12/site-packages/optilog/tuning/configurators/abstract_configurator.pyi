import abc
from _typeshed import Incomplete as Incomplete
from abc import ABC
from optilog.abstractscenario import AbstractScenario as AbstractScenario
from optilog.blackbox import BlackBox as BlackBox
from typing import Any, Callable

class TuningEntrypointType:
    """
    **NOTE: This is not a class, it is a Type Alias**

    This alias can be:
        - A Python function. If this python function is configurable it must be annotated with `@ac`
        - A PythonBlackBox (object or class)
        - A SystemBlackBox (object or class)
    """
    TYPES = BlackBox | Callable | type[BlackBox]

class TuningGlobalConfigurableType:
    """
    **NOTE: This is not a class, it is a Type Alias**

    This alias can be:
        - A Python function annotated with `@ac`
        - A PythonBlackBox (class)
        - A SystemBlackBox (class)
    """
    TYPES = Callable | type[BlackBox]
ENTRYPOINT_TYPE = TuningEntrypointType
CONFIGURABLE_TYPE = TuningGlobalConfigurableType
CONFIGURABLE_TYPE_LIST = list[CONFIGURABLE_TYPE]

class AbstractConfigurator(AbstractScenario, ABC, metaclass=abc.ABCMeta):
    SAFE_CHAR_SET: Incomplete
    AS_RUNNABLE_ID: int
    template_name: Incomplete
    entrypoint: Incomplete
    global_cfg: Incomplete
    input_data: Incomplete
    data_kwarg: Incomplete
    seed_kwarg: Incomplete
    run_obj: Incomplete
    quality_regex: Incomplete
    cost_max: Incomplete
    input_type: Incomplete
    input_type_name: Incomplete
    def __init__(self, template_name, entrypoint, global_cfg, constraints, run_obj, data_kwarg, seed_kwarg, quality_regex, input_data: Incomplete | None = None, cost_max: Incomplete | None = None) -> None: ...
    def run_raw(self, *args, **kwargs) -> None: ...
    @staticmethod
    def get_command(scenario_path: str, config: dict[str, Any], instance: str, seed: int = None):
        """Generates the execution command of the scenario.
        Only usable if the entrypoint is a BlackBox.
        
        :param scenario_path: Path to the scenario directory
        :param config: Configuration extracted from a parser
        :param instance: Instance to substitute in the blackbox 
        :param seed: Seed to substitute in the blackbox 
        """
    @staticmethod
    def get_entrypoint_kwargs(entrypoint, data_kwarg, seed_kwarg, input_data) -> None: ...
    def generate_scenario(self, out_dir: str, log: bool = True):
        """Generates all the files required by the Automatic Configurator.
        
        :param out_dir: Output directory where files will be generated.
        :param log: If True, will print a log to the console
        """
    def configure(self, config) -> None: ...
    @staticmethod
    def cast_params(cfg, params) -> None: ...
