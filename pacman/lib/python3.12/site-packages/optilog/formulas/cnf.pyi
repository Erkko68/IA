from .absformula import AbsFormula as AbsFormula
from _typeshed import Incomplete as Incomplete

class CNF(AbsFormula):
    """Creates an in-memory representation of a CNF
    (`Conjunctive Normal Form`_) formula using the DIMACS [#cnf]_ format.

    DIMACS format defines that variables start at 1, and their negation is
    the negative value (e.g -1).
    """
    ftype: str
    clauses: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def num_clauses(self) -> int:
        """
        :return: The number of clauses in the formula.
        """
    def satisfied(self, model: list[int]) -> bool | None:
        """
        Checks if a given assignment satisfies the CNF formula.

        :param model: A (possibly partial) assignment for the CNF formula.
        :return: True/False (if the assignment satisfies or not) or None if some clause is undefined (notice that the assignment can be partial).

        .. code-block:: python

            >>> cnf.clauses
            [[1,2,3],[-1,-2],[-1,-3],[-2,-3]]
            >>> cnf.satisfied([-1,2,3])
            False
            >>> cnf.satisfied([-1,-2,3])
            True

        """
    var_manager: Incomplete
    def copy(self) -> CNF:
        """
        |form-copy-description|

        :return: A copy of the CNF formula.
        """
    def add_clauses(self, clauses: list[list[int]]) -> None:
        """
        Adds a set of clauses to the CNF formula. Extends variables if necessary.

        :param clauses: The clauses to be added to the CNF formula.
        """
    def add_clause(self, clause: list[int]) -> None:
        """Adds a new clause to the CNF formula. Extends variables if necessary.

        :param clause: The clause to be added to the CNF formula.
        """
    def shuffle(self, vars: bool = True, clauses: bool = True, literals: bool = True, polarity: bool = True, seed: int = None) -> None:
        """
        Shuffles the variables, clauses, literals and polarity of the formula.
        This may be used to test the performance of encodings on a formula.

        :param vars: Whether to shuffle the variables of the formula.
        :param clauses: Whether to shuffle the order of the clauses of the formula.
        :param literals: Whether to shuffle the order of the literals of the clauses.
        :param polarity: Whether to shuffle the order of the polarity of the variables.
        :param seed: Optional seed for the shuffle

        .. code-block:: python
            
            >>> from optilog.formulas import CNF
            >>> c = CNF()
            >>> c.add_clause([4,2,5])
            >>> c.add_clause([4,2])
            >>> c.add_clause([4])
            >>> c.clauses
            >>> [[4, 2, 5], [4, 2], [4]]
            >>> c.shuffle(vars=True, clauses=False)
            >>> c.clauses
            >>> [[3, 1, 2], [3, 1], [3]]
            >>> c.shuffle(vars=False, clauses=True)
            >>> c.clauses
            >>> [[3, 1], [3], [3, 1, 2]]
        """
    def statistics(self) -> dict:
        """
        Computes:
    
        - int: The number of variables
        - int: The maximum of variable
        - int: Number of clauses of the formula
        - [(size, frequency)] The number of clauses by size
        - [(var, positive frequency, negative frequency)] The frequency of the variables
        - int: The number of literals.

        :return: Statistics of the formula

        .. code-block:: python

            >>> from optilog.formulas import CNF
            >>> c = CNF()
            >>> c.add_clause([2,3,8])
            >>> c.add_clause([10,3,1])
            >>> c.statistics()
            {'n_vars': 10, 'n_clauses': 2, 'clauses_by_size': [(3, 2)], 'frequency': [(1, 1, 0), (2, 1, 0), (3, 2, 0), (8, 1, 0), (10, 1, 0)], 'n_lits': 6}

        """
