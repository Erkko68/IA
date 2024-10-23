from .absformula import AbsFormula as AbsFormula
from _typeshed import Incomplete as Incomplete

class QCNFException(Exception):
    """Invalid QCNF operation."""

class QCNF(AbsFormula):
    E: str
    A: str
    QUANTIFIERS: Incomplete
    quantified_vars: Incomplete
    quants: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def num_clauses(self) -> int:
        """The number of clauses in the formula.

        :return: _description_
        """
    def num_quant_levels(self) -> int:
        """
        :return: The number of quantified levels within the QCNF formula.
        """
    @property
    def clauses(self) -> list[list[int]]:
        """The list of all clauses in the formula.
        """
    def is_fully_quantified(self) -> None: ...
    def quant_level(self, index) -> None: ...
    def new_var(self) -> int:
        """|form-newvar-description|

        :return: The newly created variable.
        """
    def extend_vars(self, how_many: int) -> None:
        """
        |form-extendvars-description|

        :param how_many: The number of variables to be created
        """
    def set_minimum_vars(self, minimum: int) -> None:
        """|form-setminimumvars-description|

        :param minimum: The minimum number of variables this formula should have.
        """
    def add_clause(self, clause: list[int]) -> None:
        """
        Adds a new clause to the QCNF formula. Extends variables if necessary.

        :param clause: The clause to be added to the formula.
        """
    def add_clauses(self, clauses: list[list[int]]) -> None:
        """Adds a set of clauses to the QCNF formula. Extends variables if necessary.

        :param clauses: The clauses to be added to the formula.
        """
    def get_quant_levels(self) -> None: ...
    def split_quant(self) -> None: ...
    def append_quant_set(self, quantifier: str, variables: list[int]) -> None:
        """
        Appends a new set of quantified variables to the QCNF formula. Extends variables if necessary.

        :param quantifier: Quantifier of the variables; ``e`` for existential, ``a`` for universal.
        :param variables: Variables to be quantified.
        """
    def append_quant_sets(self, sets) -> None: ...
    def copy(self) -> QCNF:
        """
        |form-copy-description|

        :return: A copy of the QCNF formula.
        """
    def filter_quantifier(self, f_quantifier) -> None: ...
