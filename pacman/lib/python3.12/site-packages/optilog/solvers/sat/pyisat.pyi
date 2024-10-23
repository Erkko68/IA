from optilog.formulas.cnf import CNF as CNF
from optilog.formulas.wcnf import WCNF as WCNF
from pathlib import Path

class PyiSAT:
    """
    PyiSAT is a standard interface to access incremental SAT solver oracles from Python.

    Not every solver implements every method of the interface. If a method is not implemented, it will throw a `NotImplementedError`.
    """
    def __init__(self, **config) -> None:
        """
        Instantiates a new SAT solver.
        Each parameter passed as a keyword argument will
        be configured on the solver.

        .. code-block:: python

            >>> from optilog.solvers.sat import Glucose41
            >>> solver = Glucose41(seed=1.0)
            >>> solver.get('seed')
            1.0
        """
    def new_var(self) -> int:
        """
        Creates a new variable and returns its DIMACS id.
        The returned value will be `PyiSAT.max_var() + 1`.

        :return: The created variable.

        .. code-block:: python

            >>> from optilog.solvers.sat import Glucose41
            >>> solver = Glucose41()
            >>> solver.new_var()
            1
            >>> solver.new_var()
            2
        """
    def add_clause(self, clause: list[int]):
        """
        Adds a new clause to this solver.

        :param clause: The clause to be added to the solver.

        .. code-block:: python

            >>> from optilog.solvers.sat import Glucose41
            >>> solver = Glucose41()
            >>> solver.add_clause([1,2,3])

        """
    def load_cnf(self, path: str | Path, return_complete: bool = False) -> CNF:
        """
        This method can be used to load CNF formulas directly into the solver.

        By default, nothing is returned.
        If you would like the parsed formula to be returned, pass the argument return_complete=True

        :param path: Path to the cnf file
        :param bool return_complete: If true, returns loads the formula into the solver and returns it

        .. code-block:: python

            >>> from optilog.solvers.sat import Glucose41
            >>> solver = Glucose41()
            >>> solver.load_cnf('./path/to/file')

        """
    def load_wcnf(self, path: str | Path, return_complete: bool = False) -> WCNF:
        """
        This method can be used to load the HARD clauses of the formulas directly into the solver.

        By default, only the SOFT clauses of the formula are returned. This is implemented for performance reasons, as many formulas contain a very large set of hard clauses but small set of soft ones.
        If you would like the entire formula to be returned, pass the argument return_complete=True

        :param path: Path to the wcnf file
        :param return_complete: If false, a formula with only the soft clauses are returned. If true, returns the complete formula.

        .. code-block:: python

            >>> from optilog.solvers.sat import Glucose41
            >>> solver = Glucose41()
            >>> solver.load_wcnf('./path/to/file')

        """
    def add_clauses(self, clauses: list[list[int]]):
        """
        Adds a set of clauses to this solver.

        :param clauses: The set of clauses to be added to the solver.

        .. code-block:: python

            >>> from optilog.solvers.sat import Glucose41
            >>> solver = Glucose41()
            >>> solver.add_clauses([[1,2,3],[-1,-2],[-1,-3],[-2,-3]])

        """
    def max_var(self) -> int:
        """
        :return: The maximum DIMACS variable id in the solver.

        .. code-block:: python

            >>> from optilog.solvers.sat import Glucose41
            >>> solver = Glucose41()
            >>> solver.add_clauses([[1,2,3],[-1,-2],[-1,-3],[-2,-3]])
            >>> solver.max_var()
            3
        """
    def num_clauses(self) -> int:
        """
        :return: Returns the number of clauses within the solver.

        .. code-block:: python

            >>> from optilog.solvers.sat import Glucose41
            >>> solver = Glucose41()
            >>> solver.add_clauses([[1,2,3],[-1,-2],[-1,-3],[-2,-3]])
            >>> solver.num_clauses()
            4

        .. warning::
            The number of clauses in the solver may not correspond with the number of clauses inserted into the solver.
            The solver may freely discard redundant clauses or not take into account the number of unitary clauses, among other considerations.
            The exact behavior is solver dependent.
        """
    def set_conf_budget(self, budget: int):
        """
        Sets the maximum number of conflicts allowed by the budget, used by the methods :py:meth:`solve_limited` or :py:meth:`solve_hard_limited`.
        A :py:meth:`solve` call will reset all budgets.
        A call with budget `0` or `1` will reset the configuration budget.
        The configuration budget is kept in between calls to :py:meth:`solve_limited` or :py:meth:`solve_hard_limited` and is not automatically reset.
        Meaning that if you want to run your each solve_limited call with a limit of 1000, you will have to manually reapply the budget in between calls.

        :param budget: The budget for the number of conflicts.

        .. code-block:: python

            >>> from optilog.solvers.sat import Glucose41
            >>> solver = Glucose41()
            >>> solver.set_conf_budget(100)
        """
    def set_prop_budget(self, budget: int):
        """
        Sets the maximum number of propagations allowed by the budget, used by the methods :py:meth:`solve_limited` or :py:meth:`solve_hard_limited`.
        A :py:meth:`solve` call will reset all budgets.
        A call with budget `0` or `1` will reset the configuration budget.
        The configuration budget is kept in between calls to :py:meth:`solve_limited` or :py:meth:`solve_hard_limited` and is not automatically reset.
        Meaning that if you want to run your each solve_limited call with a limit of 1000, you will have to manually reapply the budget in between calls.

        :param budget: the budget for the number of propagations.

        .. code-block:: python

            >>> from optilog.solvers.sat import Glucose41
            >>> solver = Glucose41()
            >>> solver.set_prop_budget(5000)

        """
    def solve(self, assumptions: list[int] | None = None) -> bool | None:
        """
        Solves the formula with an optional list of assumptions (list of literals that are forced to be true).

        :param assumptions: List of assumptions.
        :return: Returns True if all variables are assigned and no contradiction is found.
            Returns False, if the formula is proven to be unsatisfiable.
            Otherwise, returns None if all decision variables are assigned and no contradiction is found.

        .. code-block:: python

            >>> from optilog.solvers.sat import Glucose41
            >>> solver = Glucose41()
            >>> solver.add_clauses([[1,2,3],[-1,-2],[-1,-3],[-2,-3]])
            >>> solver.solve([-1])
            True
            >>> solver.solve([1, 3])
            False
            >>> solver.add_clause([4, 5])
            >>> solver.set_decision_var(4, False)
            >>> solver.set_decision_var(5, False)
            >>> solver.solve()
            None
            >>> solver.model()
            [-1, 2, -3]

        """
    def solve_limited(self, assumptions: list[int] | None = None) -> bool | None:
        """
        Stops the solving process if the budget is already exhausted at the beginning of the next restart.

        If no budget has been set, this call is equivalent to the :py:meth:`solve` method.

        :param assumptions: List of assumptions.
        :return: Returns the same as method :py:meth:`solve` and additionally returns None if the budget is exhausted.
        """
    def solve_hard_limited(self, assumptions: list[int] | None = None) -> bool | None:
        """
        Stops the solving process if the budget is already exhausted.

        If no budget has been set, this call is equivalent to the :py:meth:`solve` method.

        :param assumptions: List of assumptions.
        :return: Returns the same as method :py:meth:`solve` and additionally returns None if the budget is exhausted.
        """
    def propagate(self, assumptions: list[int] | None = None, remove_assumptions: bool = True) -> tuple[bool, list[int]]:
        """
        Applies unit propagation with an optional list of assumptions.

        :param assumptions: The list of assumptions.
        :param remove_assumptions: If True, the assumptions are removed from the list of propagated literals.
        :return: Returns True if the solver finds no contradiction, and the list of assigned literals.
            Returns False if the solver finds a contradiction and an empty list.
        """
    def model(self) -> list[int]:
        """
        :return: Returns a model for the decision variables that satisfies the formula.

        .. code-block:: python

            >>> from optilog.solvers.sat import Glucose41
            >>> solver = Glucose41()
            >>> solver.add_clauses([[1,2,3],[-1,-2],[-1,-3],[-2,-3]])
            >>> solver.solve([-1])
            >>> solver.model()
            [-1, 2, -3]
        """
    def is_full_model(self) -> bool:
        """
        :return: True if model assigns all the variables. False otherwise.
        """
    def core(self) -> list[int]:
        """
        If the solver returns False, computes a subset of the assumptions that form a core of unsatisfiability.

        :return: A core of unsatisfiability.

        .. code-block:: python

            >>> from optilog.solvers.sat import Glucose41
            >>> solver = Glucose41()
            >>> solver.add_clauses([[1,2,3], [-1,-2], [-1,-3], [-2,-3]])
            >>> solver.solve([1,2,3])
            False
            >>> solver.core()
            [2, 1]

        """
    def get_polarity(self, variable: int) -> bool:
        """
        Returns the preferred value for the given variable when the solver makes a decision.
        This value may be updated during the resolution process by phase saving policies.
        The default preferred value depends on the solver implementation.

        :param variable: The variable id.
        :return: The preferred polarity value for a variable.

        .. code-block:: python

            >>> from optilog.solvers.sat import Glucose41
            >>> solver = Glucose41()
            >>> solver.add_clauses([[1,2,3],[-1,-2],[-1,-3],[-2,-3]])
            >>> solver.get_polarity(1)
            False

        """
    def set_polarity(self, variable: int, polarity: bool):
        """
        Forces the solver to set the given polarity when deciding the variable.

        :param variable: The variable id.
        :param polarity: The preferred value for the variable.
        """
    def unset_polarity(self, variable: int):
        """
        Disables the polarity set by the method :py:meth:`set_polarity`.

        :param variable: The variable id.
        """
    def value(self, variable: int) -> bool | None:
        """
        :param variable: The variable id.
        :return: Returns the value assigned to the given variable.             If the variable is unassigned, returns None (meaning undefined).

        .. code-block:: python

            >>> from optilog.solvers.sat import Glucose41
            >>> solver = Glucose41()
            >>> solver.add_clauses([[1,2,3],[-1,-2],[-1,-3],[-2,-3]])
            >>> v = solver.value(1)
            >>> print(v)
            None
            >>> solver.solve()
            True
            >>> solver.model()
            [-1, 2, -3]
            >>> solver.value(1), solver.value(2), solver.value(3)
            (False, True, False)

        """
    def interrupt(self) -> None:
        """
        Interrupts the resolution process.
        This interruption will happen at the beginning of the next restart.
        """
    def clear_interrupt(self) -> None:
        """
        Clears an interruption.
        The interruption must be cleared before calling ``solve`` like methods.
        **Not all the solvers force to clear the interruptions.**
        """
    def num_conflicts(self) -> int:
        """
        :return: Returns the number of conflicts detected during the solving process.
        """
    def set_decision_var(self, variable: int, dec: bool = False):
        """
        :param variable: The variable to be set as decision variable.
        :param dec: If dec is True, sets the variable as a decision variable.
            If dec is False the variable is not a decision variable.
            In a new solver, by default all variables are decision variables.
        """
    def trace_proof(self, file_path: str):
        """

        Dumps the proof of the resolution process ot the specified file_path.
        This method must be called before executing any of the ``solve`` like methods.
        If this method is called after any ``solve`` like call an exception will arise.

        :param str file_path: The file path where the proof is going to be written.

        .. warning::

            Solver interfaces are not homogeneous so this specification may not be entirely suitable.
        """
    def learnt_clauses(self) -> list[list[int]]:
        """
        :return: Returns the list of learnt clauses. Only clauses of size > 1 are guaranteed.
        """
    def set_static_heuristic(self, heuristic: list[int]):
        """
        Sets an static heuristic. The heuristic is described as a list of pairs of variables and values.

        If all the variables in the heuristic have been decided, the solver applies the regular procedure.

        If an heuristic is active then its polarities take precedence over the ones assigned by the method :py:meth:`set_polarity`.

        :param heuristic: A list of literals.
        """
    def enable_static_heuristic(self, enable: bool = True):
        """
        :param enable: Enables or disables the static heuristic set by method :py:meth:`set_static_heuristic`
        """
    def set_incremental(self) -> None:
        '''
        Enables the `incremental` mode in solvers of the `Glucose` family.
        This mode enables changes in the LBD heuristic that may help improve performance in incremental sat solving.
        For more details see "Improving Glucose for Incremental SAT Solving with Assumptions: Application to MUS Extraction".
        '''
    def clone(self) -> PyiSAT:
        """
        Creates a clone of the PyiSAT solver.
        The new solver will have all the same internal state as the original solver.
        :return: Cloned solver
        """
    def get_trail(self, level: int) -> list[int]:
        """
        Obtain the literals of the solver's trail on the specified decision level.

        :param level: Trail level
        :return: Solver's trail
        """
    def unsat_state(self) -> bool:
        """
        :return: Returns True if the solver is in an unsat state. An unsat state means that the current formula is unsatisfiable.
        """
    @staticmethod
    def get_config() -> dict[str, dict]:
        """
        :return: Returns a dictionary with all the configurable parameters of the solver.
           Each parameter has defined its type (*int*, *float*, *bool* or *categorical*), its domain and its default value.

        .. code-block:: python

            >>> from optilog.solvers.sat import Glucose41
            >>> config = Glucose41.get_config()
            >>> config
            {
            'params': {
                'gc-frac': {
                    'type': 'real'
                    'domain': [0.001, 3.4e+38],
                    'default': 0.2,
                },
                'rinc': {
                    'type': 'integer'
                    'domain': [2, 2147483647],
                    'default': 2,
                },
                'rnd-init': {
                    'type': 'bool'
                    'domain': [True, False],
                    'default': False,
                }
                ........
            }
            }
        """
    def set(self, key: str, value: type[int | float | bool | str]):
        """
        Sets the value for a given parameter.

        :param key: Parameter name.
        :param value: Parameter value.
        :type value: int or float or bool or str
        :raises: KeyError: if ``param`` is not found inside the solver.

        .. code-block:: python

            >>> from optilog.solvers.sat import Glucose41
            >>> solver = Glucose41()
            >>> solver.set('seed', 1.0)

        """
    def get(self, key: str) -> int | float | bool | str:
        """
        Returns the value assigned to a given parameter name.

        :param key: Parameter name.
        :return: Parameter value.
        :raises: KeyError: if ``param`` is not found inside the solver.

        .. code-block:: python

            >>> from optilog.solvers.sat import Glucose41
            >>> solver = Glucose41()
            >>> solver.set('seed', 1.0)
            >>> solver.get('seed')
            1.0

        """
