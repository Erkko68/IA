import abc
from _typeshed import Incomplete as Incomplete
from abc import ABC
from typing import IO

class NewLineStreamer:
    stream: Incomplete
    printed_before: bool
    def __init__(self, stream) -> None: ...
    def print(self, *args) -> None: ...

class CNFException(Exception):
    """Invalid CNF operation."""

class VarManager:
    """
    A VarManager represents a pool of variables for a specific problem. It can be used
    to create variables and to keep track of the ones already created. Each variable
    created by a VarManager is assigned an identifier according to the DIMACS format,
    starting at 1.

    The VarManager class is used by  :ref:`CNF <CNF-formula>`, :ref:`WCNF <WCNF-formula>`,
    and :ref:`QCNF <QCNF-formula>` formulas.
    """
    mapping: Incomplete
    inverse_mapping: Incomplete
    def __init__(self) -> None: ...
    def copy(self) -> VarManager:
        """
        :return: A copy of the VarManager
        """
    def extend_vars(self, how_many: int) -> None:
        """|form-extendvars-description|

        :param how_many: The number of variables to be created.
        """
    def set_minimum_vars(self, minimum: int) -> None:
        """|form-setminimumvars-description|

        :param minimum: The minimum number of variables this formula should have.
        """
    def add_var(self, lit: Expression) -> int:
        """
        Creates a new DIMACS variable and associates it to the Expression.

        Used to connect variables created with the :ref:`Modelling module <modelling-module>`
        with an existing variable pool.


        :param lit: The expression to associate
        :return: The associated DIMACS literal
        """
    def max_var(self) -> int:
        """
        :return: The maximum DIMACS variable id in the formula.
        """
    def new_var(self) -> int:
        """|form-newvar-description|

        :return: The newly created variable.
        """
    def get_lit(self, lit: Expression, insert: bool = True) -> int:
        """
        Returns the DIMACS literal associated with the Expression.

        The `lit` parameter can be either an instance of a `Bool`, or its
        negation (an instance of a `Not(Bool)` expression). It also accepts
        nested negations, as long as the last element of the tree is a Bool.
        If the argument `insert` is True and no DIMACS literal is associated
        with the Expression, a new literal is created and associated.

        :param lit: The expression to lookup
        :param insert: Whether to create a new variable if the Expression has not been found.
        :raises ValueError: if the expression is not unitary
        :raises KeyError: if the literal is not found and the parameter `insert` is set to False
        :return: The associated DIMACS literal
        """
    def decode_dimacs(self, model) -> None: ...

class AbsFormula(ABC, metaclass=abc.ABCMeta):
    var_manager: Incomplete
    def __init__(self, var_manager: Incomplete | None = None) -> None: ...
    def to_dimacs(self, clause) -> None: ...
    def extend_vars(self, how_many: int) -> None:
        """|form-extendvars-description|

        :param how_many: The number of variables to be created.
        """
    def set_minimum_vars(self, minimum) -> None:
        """|form-setminimumvars-description|

        :param minimum: The minimum number of variables this formula should have.
        """
    def max_var(self) -> int:
        """
        :return: The maximum DIMACS variable id in the formula.
        """
    def decode_dimacs(self, model) -> None: ...
    def header(self) -> list[str]:
        """
        :return: A list of strings representing comments on the formula.
        """
    def new_var(self) -> int:
        """|form-newvar-description|

        :return: The newly created variable.
        """
    def write_dimacs(self, stream: IO = ..., *args, **kwargs):
        """
        Prints to the given stream the formula into DIMACS format.

        :param stream: The stream where to print the formula.
        """
    def write_dimacs_file(self, file_path: str, *args, **kwargs) -> None:
        """Writes to the given file path the formula into DIMACS format.

        :param file_path: The file path where to write the formula.

        .. code-block:: python

            >>> cnf.write_dimacs_file('example')
        """
    def add_comment(self, comment: str | list[str]) -> None:
        """
        Adds a comment to the header of the formula.
        These comments will be displayed in the DIMACS format with
        the character ``c`` prepended, hence the DIMACS parser will ignore them.

        :param comment: Comment to append to the header.

        .. code-block:: python

            >>> form.add_comment('a simple test formula')
        """
