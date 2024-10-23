from _typeshed import Incomplete as Incomplete
from collections.abc import Generator

class PBEncoderTransformer:
    comparator: Incomplete
    pbenc: Incomplete
    amoenc: Incomplete
    amkenc: Incomplete
    @staticmethod
    def run(input_data) -> Generator[Incomplete, Incomplete, None]: ...
    @staticmethod
    def get_params(args) -> None: ...
    @staticmethod
    def get_conditions(args) -> None:
        """
        # Conditionals:
        child_name | condition [&&,||] condition ...

        # Condition Operators:
        # parent_x [<, >] parent_x_value (if parameter type is ordinal, integer or real)
        # parent_x [==,!=] parent_x_value (if parameter type is categorical, ordinal or integer)
        # parent_x in {parent_x_value1, parent_x_value2,...}
        """
