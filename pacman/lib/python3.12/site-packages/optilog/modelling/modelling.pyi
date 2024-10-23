from .expressions import *
from .problem import Problem as Problem
from _typeshed import Incomplete as Incomplete

class TruthTable:
    """
    A class to represent and handle truth tables for Boolean expressions. Can print the TruthTable on the terminal.
    """
    problems: Incomplete
    def __init__(self, *problems: Problem) -> None: ...
    def add_problem(self, problem: Problem): ...
    def print(self, variable_order: Incomplete | None = None) -> None: ...

def resolve(c1, c2) -> None: ...
