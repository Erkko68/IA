from _typeshed import Incomplete as Incomplete
from typing import Any

class SMACParser:
    """Instantiates the parser of the configuration
    """
    input_path: Incomplete
    def __init__(self, input_path: str) -> None:
        """
        :param input_path: Path to the logs of the workers of SMAC
        """
    def get_configs(self, raise_if_incomplete: bool = True) -> dict[str, dict[str, Any]]:
        """Get the best config from the output of the SMAC workers.
        Note that SMAC provides a configuration for each worker.

        :param raise_if_incomplete: Wether to raise an exception if no config is found
        :return: A dictionary with the best configuration
        """
    def save_configs(self, output_path: str, scenario_path: str, name: str):
        """Save the best configurations in a json file and an executable wrapper.

        :param output_path: The directory where the files will be written to
        :param scenario_path: The path to the configuration scenario
        :param name: Name for the configuration that will be used in the generated files
        """
