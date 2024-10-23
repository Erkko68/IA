from .expressions import *
from _typeshed import Incomplete as Incomplete
from optilog.formulas import CNF as CNF

class ImplicationTranslator:
    """Translates a set of constraints that may contain implications
    to another set of constraints without any implication.

    In particular, this class handles implications and double-implications.
    """
    exprs: Incomplete
    def __init__(self, exprs: list[Expression]) -> None: ...
    def encode(self) -> list[Expression]: ...

class CNFPlusTranslator:
    """Converts a formula with ORs, ANDs, NOT and PBs into a CNF
    with Boolean variables, other CNFs and PBs (CNF+).
    """
    exprs: Incomplete
    def __init__(self, exprs: list[Expression]) -> None: ...
    def encode(self) -> list[Expression]: ...
    def get_reify_registry(self) -> None: ...
    def get_pb_registry(self) -> None: ...

class PBNorm:
    """Static class (inspired in the beautifulness of Java)
    that we use to group all the functionality related to PB Normalization.
    """
    @staticmethod
    def order_sides(pb_expr) -> None:
        """Orders the sides of a LessThan expression.

        All the coefficients are placed in the Left Hand Side (LHS)
        and all the single integers into the Right Hand Side (RHS).

        The integers in the RHS are simplified.

        We just negate the subexpression to move it to the opposite side.

        :param pb_expr: LessThan expression to convert
        :type pb_expr: LessThan
        :return: A new instance of LessThan with the transformation applied
        :rtype: LessThan
        """
    @staticmethod
    def remove_neg_coeffs(pb_expr) -> None:
        """Removes the negative coefficients of a LessThan expression.

        For each negative coefficient (which at this point will be part of a Multiply sub-expression),
        we convert it to positive by negating its literal and updating the RHS of the expression.

        TODO: How do we update the RHS and WHY?

        :param pb_expr: LessThan expression to convert
        :type pb_expr: LessThan
        :return: A new instance of LessThan with the transformation applied
        :rtype: LessThan
        """
    @staticmethod
    def group_coeffs(pb_expr) -> None:
        """Group the coefficient of the LHS of a LessThan expression.

        :param pb_expr: LessThan expression to convert
        :type pb_expr: LessThan
        :return: A new instance of LessThan with the transformation applied
        :rtype: LessThan
        """
    @staticmethod
    def sort_coeffs(pb_expr) -> None:
        """Sorts the coefficients of the LHS of a LessThan expression in descending order.

        TODO: Break ties alphabetically. This could be useful for the cache!

        :param pb_expr: LessThan expression to convert
        :type pb_expr: LessThan
        :return: A new instance of LessThan with the transformation applied
        :rtype: LessThan
        """
    @staticmethod
    def trim_coeffs(pb_expr) -> None:
        """Trims all the coefficients in the LHS of a LessThan expression.

        In particular, all the coefficients that are greater than the bound
        are trimmed to the bound.

        :param pb_expr: LessThan expression to convert
        :type pb_expr: LessThan
        :return: A new instance of LessThan with the transformation applied
        :rtype: LessThan
        """
    @staticmethod
    def simplify_coeffs(pb_expr) -> None: ...
    @staticmethod
    def is_trivially_false(pb_expr) -> None: ...
    @staticmethod
    def is_trivially_true(pb_expr) -> None: ...

class Clause:
    lits: Incomplete
    def __init__(self, lits: list[int]) -> None: ...

class InnerCNF:
    clauses: Incomplete
    def __init__(self, clauses: list[list[int]]) -> None: ...

class CNFPlusToCNFTranslator:
    """Translates a set of expressions in CNF+ to a single CNF."""
    cnf: Incomplete
    exprs: Incomplete
    def __init__(self, cnf: CNF, exprs: list[Expression]) -> None: ...
    def encode(self) -> CNF: ...
