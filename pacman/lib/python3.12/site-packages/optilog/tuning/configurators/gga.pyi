from .abstract_configurator import AbstractConfigurator as AbstractConfigurator, CONFIGURABLE_TYPE_LIST as CONFIGURABLE_TYPE_LIST, ENTRYPOINT_TYPE as ENTRYPOINT_TYPE
from _typeshed import Incomplete as Incomplete
from optilog.blackbox import ExecutionConstraints as ExecutionConstraints

class GGAScenario(AbstractConfigurator):
    """Handles the creation of all the needed files to use the GGA AC tool over a configurable function.
    """
    scenario_kwargs: Incomplete
    default_as_elite: Incomplete
    eval_time_limit: Incomplete
    seed: Incomplete
    def __init__(self, entrypoint: ENTRYPOINT_TYPE, constraints: ExecutionConstraints, run_obj: str, seed: int, cost_min: int, cost_max: int, global_cfg: CONFIGURABLE_TYPE_LIST = [], eval_time_limit: int = None, data_kwarg: str = None, seed_kwarg: str = None, input_data: str = None, quality_regex: str = None, default_as_elite: bool = False, **scenario_kwargs) -> None:
        """
        :param entrypoint: |entrypoint-param|
        :param global_cfg: |calls-params|
        :param input_data: |input-param|
        :param constraints: |constraints-param|
        :param run_obj: |run-param|
        :param data_kwarg: |data-param|
        :param seed_kwarg: |seed-param|
        :param quality_regex: |quality-param|
        :param seed: Seed used by GGA on every instance
        :param cost_min: Minimum possible cost of the algorithm
        :param cost_max: Maximum possible cost of the algorithm
        :param eval_time_limit: Time limit for each evaluation in seconds (defaults to the wall time limit constraint)
        :param \\*\\*scenario_kwargs:             GGA scenario parameters as they appear in             GGA's official documentation. These parameters             will be directly written to the GGA scenario file.             GGA documentation can be accessed through the `official website of LOG <http://ulog.udl.cat/software/>`_.
        """
    @staticmethod
    def get_instances_directory(scenario_dir) -> None: ...
