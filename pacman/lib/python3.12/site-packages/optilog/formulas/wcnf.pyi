from .absformula import AbsFormula as AbsFormula
from _typeshed import Incomplete as Incomplete
from typing import IO

class WCNFException(Exception):
    """Invalid WCNF operation."""

class WCNF(AbsFormula):
    """
    Creates an in-memory representation of a WCNF
    (Weighted Conjunctive Normal Form) formula.
    """
    INF_WEIGHT: int
    hard_clauses: Incomplete
    soft_clauses: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def sum_soft_weights(self) -> int | float:
        """
        :return: The sum of all soft weights.
        """
    def num_clauses(self) -> int:
        """
        :return: The number of clauses (soft and hard) in the formula.
        """
    def top_weight(self) -> int | float:
        """
        :return: The weight for hard clauses = ``WCNF.sum_soft_weights + 1``.
        """
    def feasible(self, model: list[int]) -> bool | None:
        """
        Checks if a given assignment satisfies the hard constraints of a WCNF formula.
        
        :param assignment: A (possibly partial) assignment for the WCNF formula.
        :return: True/False (if the assignment satisfies or not) or None if some hard clause is undefined (notice that the assignment can be partial).

        .. code-block:: python

            >>> wcnf = WCNF()
            >>> wcnf.add_clauses([[1,2,3],[-1,-2],[-1,-3],[-2,-3]])
            >>> wcnf.add_clauses([[-1], [-2]], weight=1)
            >>> wcnf.feasible([-1,2,3])
            >>> False
            >>> wcnf.feasible([-1,-2,3])
            >>> True
            >>> wcnf = WCNF()
            >>> wcnf.add_clauses([[1,2,3], [-1,-2]])
            >>> wcnf.add_clauses([[-1]], weight=1)
            >>> wcnf.feasible([3])
            >>> None

        """
    var_manager: Incomplete
    def copy(self) -> WCNF:
        """|form-copy-description|

        :return: A copy of the WCNF formula.
        """
    def add_clauses(self, clauses: list[list[int]], weight: int = ...):
        """
        Adds a set of clauses of same weight to the WCNF formula. Extends variables if necessary.

        :param clause: The clause to be added to the formula.
        :param weight: |wcnf-opt-weight-description|

        .. code-block:: python

            >>> wcnf = WCNF()
            >>> wcnf.add_clauses([[-1],[-2],[-3]], 1)
            >>> wcnf.add_clauses([[1,2],[2,3]])
            >>> print(wcnf)
            p wcnf 3 5 4
            c ===== Hard Clauses =====
            4 1 2 0
            4 2 3 0
            c ===== Soft Clauses (Sum weights: 3) =====
            1 -1 0
            1 -2 0
            1 -3 0

        """
    def add_clause(self, clause: list[int], weight: int = ...):
        """
        Adds a new clause to the WCNF formula. Extends variables if necessary.

        :param clause: The clause to be added to the formula.
        :param weight: |wcnf-opt-weight-description|

        .. code-block:: python

            >>> wcnf = WCNF()
            >>> wcnf.add_clause([1,2,3])
            >>> wcnf.add_clause([1], 1)
            >>> wcnf.add_clause([2], 1)
            >>> wcnf.add_clause([3], 1)
            >>> print(wcnf)
            p wcnf 3 4 4
            c ===== Hard Clauses =====
            4 1 2 3 0
            c ===== Soft Clauses (Sum weights: 3) =====
            1 1 0
            1 2 0
            1 3 0
        """
    def cost(self, assignment: list[int], inverse: bool = False) -> int | float:
        """
        :param assignment: A (possibly partial) assignment for the WCNF formula.
        :param inverse: *Inverse* parameter.
        :return: Returns the aggregated weight of soft clauses violated or undefined by the assignment.             If ``inverse`` parameter is True, returns the aggregated weight of the soft clauses satisfied by the assignment.

        .. code-block:: python
        
            >>> wcnf = WCNF()
            >>> wcnf.add_clause([1,2,3])
            >>> wcnf.add_clauses([[-1],[-2],[-3]], 1)
            >>> wcnf.cost([1,2])
            3
            >>> wcnf.cost([1,2], inverse=True)
            0
            >>> wcnf.cost([1,-2])
            2
            >>> wcnf.cost([1,-2], inverse=True)
            1

        """
    def write_dimacs(self, stream: IO = ..., format: str = 'classic'):
        '''
        Prints to the given stream the formula into DIMACS format.

        :param stream: The stream where to print the formula.
        :param format: The wcnf format to output. Options are "classic" or "simplified"
        '''
    def write_dimacs_file(self, file_path: str, format: str = 'classic') -> None:
        '''Writes to the given file path the formula into DIMACS format.

        :param file_path: The file path where to write the formula.
        :param format: The wcnf format to output. Options are "classic" or "simplified"

        .. code-block:: python

            >>> cnf.write_dimacs_file(\'example\')
        '''
    def shuffle(self, vars: bool = True, soft_clauses: bool = True, hard_clauses: bool = True, literals: bool = True, polarity: bool = True, seed: int = None) -> None:
        """
        Shuffles the variables, clauses, literals and polarity of the formula.
        This may be used to test the performance of encodings on a formula.

        :param vars: Whether to shuffle the variables of the formula.
        :param soft_clauses: Whether to shuffle the order of the soft_clauses of the formula.
        :param hard_clauses: Whether to shuffle the order of the hard_clauses of the formula.
        :param literals: Whether to shuffle the order of the literals of the clauses.
        :param polarity: Whether to shuffle the order of the polarity of the variables.
        :param seed: Optional seed for the shuffle

        .. code-block:: python

            >>> w = WCNF()
            >>> w.add_clause([4,2,5])
            >>> w.add_clause([4,5])
            >>> w.add_clause([4,-2,3], weight=3)
            >>> w.add_clause([4,-2,5], weight=5)
            >>> w.hard_clauses
            [[4, 2, 5], [4, 5]]
            >>> w.soft_clauses
            [(3, [4, -2, 3]), (5, [4, -2, 5])]
            >>> w.shuffle()
            >>> w.hard_clauses
            [[1, 4, 5], [1, 5]]
            >>> w.soft_clauses
            [(5, [1, -4, 5]), (3, [1, -4, 2])]
        """
