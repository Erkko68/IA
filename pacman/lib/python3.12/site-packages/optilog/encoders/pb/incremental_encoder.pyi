from _typeshed import Incomplete as Incomplete

class FullIncrEncoder:
    """Full incremental encoder.

    This encoder can be used to enforce bounds to a set of variables.
    Changes in variables' values are also reflected into out vars.
    """
    max_var: Incomplete
    i_const: Incomplete
    config: Incomplete
    bound: Incomplete
    aux_var: Incomplete
    Pb2cnf: Incomplete
    already_init: bool
    def __init__(self, literals: list[int], bound: int = 1, weights: list[int] | None = None, max_var: int | None = None) -> None:
        """
        :param literals: List of literals.
        :param bound: Bound to be enforced (normally len(literals)).
        :param weights: Weights of literals (default all 1).
        :param max_var: Max var already used in formula (default max(literals)).
        """
    def get_init_clauses(self) -> tuple[int, list[list[int]]]:
        """Get initial clauses.

        :raises RuntimeError: when called twice.

        :returns: maximum variable used and set of initial clauses.
        """
    def leq(self, new_bound: int) -> list[list[int]]:
        """Encode (<= new_bound)

        :param new_bound: new_bound to be encoded

        :returns: list of clauses.
        """
    def geq(self, new_bound: int) -> list[list[int]]:
        """Encode (=> new_bound)

        :param new_bound: new_bound to be encoded

        :returns: list of clauses.
        """
    def lt(self, new_bound: int) -> list[list[int]]:
        """Encode (< new_bound)

        :param new_bound: new_bound to be encoded

        :returns: list of clauses.
        """
    def gt(self, new_bound: int) -> list[list[int]]:
        """Encode (> new_bound)

        :param new_bound: new_bound to be encoded

        :returns: list of clauses.
        """
    def eq(self, new_bound: int) -> list[list[int]]:
        """Encode (== new_bound)

        :param new_bound: new_bound to be encoded

        returns list of clauses.
        """
    def diff(self, new_bound: int) -> list[list[int]]:
        """Encode (!= new_bound)

        :param new_bound: new_bound to be encoded

        returns list of clauses.
        """
    def leq_out_vars(self) -> list[tuple[int, list[list[int]]]]:
        """Return all out vars/clauses related with leq encoding.

        Lower level function.

        returns list of (bound, clauses)
        """
    def geq_out_vars(self) -> list[tuple[int, list[list[int]]]]:
        """Return all out vars/clauses related with geq encoding.

        Lower level function.

        :returns: list of (bound, clauses)
        """

class IncrementalEncoder:
    """
    This class provides access to the most common incremental Pseudo-Boolean and Cardinality constraint encodings.
    The user can refine the bound once the constraint has been encoded.
    *IncrementalEncoder* only supports *at most k* constraints.
    
    List of available Pseudo-Boolean encodings:

    - Adder networks :cite:p:`een2006translating`
    - Sequential weigh counters :cite:p:`holldobler2012compact`

    List of available Cardinality encodings:

    - BDD :cite:p:`een2006translating` :cite:p:`abio2011bdds`
    - Cardinality network :cite:p:`asin2009cardinality`
    - Totalizer :cite:p:`bailleux2003efficient`
    """
    literals: Incomplete
    bound: Incomplete
    weights: Incomplete
    max_var: Incomplete
    encoding: Incomplete
    def __init__(self, literals: list[int], bound: int = 1, weights: list[int] | None = None, max_var: int | None = None, encoding: str = 'best') -> None:
        """
        :param literals: List of literals
        :param bound: Bound of constraint (i.e. :math:`\\leq bound`)
        :param weights: Weight of constraint (default None, unweighted)
        :param max_var: Maximum variable used so far (default: max(literals))
        :param encoding: Encoding (see :py:meth:`pb_encodings` and :py:meth:`card_encodings`). Defaults to *best*
        """
    def get_init_clauses(self) -> tuple[int, list[list[int]]]:
        """
        Obtain the initial clauses that encode the constraint specified
        in the constructor of the class.

        :return: Returns the maximum auxiliary variable used so far and the list of clauses

        .. code-block:: python

            >>> from optilog.encoders.pb import IncrementalEncoder
            >>> enc = IncrementalEncoder([1,2,3], 3)
            >>> enc.get_init_clauses()
            (6, [[-3, 4], [-2, 4], [-3, -2, 5], [-1, 4], [-3, -1, 5], [-2, -1, 5], [-3, -2, -1, 6]])
        """
    @staticmethod
    def init(literals: list[int], bound: int = 1, weights: list[int] | None = None, max_var: int | None = None, encoding: str = 'best') -> tuple['IncrementalEncoder', int, list[list[int]]]:
        """
        Utility method that returns an *IncrementalEncoder* instance and the result of calling :py:meth:`get_init_clauses`.

        :param literals: List of literals
        :param bound: Bound of constraint (i.e. *≤ bound*)
        :param weights: Weight of constraint (default None, unweighted)
        :param max_var: Maximum variable used so far (default: max(literals))
        :param encoding: Encoding (see :py:meth:`pb_encodings` and :py:meth:`card_encodings`). Defaults to *best*
        :return: *IncrementalEncoder* instance, maximum auxiliary variable used so far and list of clauses

        .. code-block:: python

            >>> from optilog.encoders.pb import IncrementalEncoder
            >>> IncrementalEncoder.init([1,2,3], 3)
            (<IncrementalEncoder object>, 6, [[-3, 4], [-2, 4], [-3, -2, 5], [-1, 4], [-3, -1, 5], [-2, -1, 5], [-3, -2, -1, 6]])

        """
    def extend(self, new_bound: int) -> tuple[int, list[list[int]]]:
        """
        Refine the original constraint incrementally.

        .. warning:: This method can only be used after calling the :py:meth:`init` or :py:meth:`get_init_clauses` methods.

        :param new_bound: Refined bound
        :return: maximum auxiliary variable used so far and list of clauses that encode the refined constraint

        .. code-block:: python

            >>> from optilog.encoders.pb import IncrementalEncoder
            >>> enc, max_var, C = IncrementalEncoder.init([1,2,3], 3)
            >>> enc.extend(2)
            (6, [[-6]])
        """
    @staticmethod
    def pb_encodings() -> list[str]:
        """
        :return: A list of strings with all the available encodings for Pseudo-Boolean constraints.

        .. code-block:: python

            >>> from optilog.encoders.pb import IncrementalEncoder
            >>> IncrementalEncoder.pb_encodings()
            ['best', 'seqcounter', 'adder']
        """
    @staticmethod
    def get_amo_encodings() -> list[str]:
        """
        :return: A list of strings with all the available At Most One constraints.

        .. code-block:: python

            >>> from optilog.encoders.pb import IncrementalEncoder
            >>> IncrementalEncoder.get_amo_encodings()
            ['best', 'nested', 'bdd', 'bimander', 'commander', 'kproduct', 'binary', 'pairwise', 'totalizer']
        """
    @staticmethod
    def card_encodings() -> list[str]:
        """
        :return: A list of strings with all the available encodings for Cardinality constraints.

        .. code-block:: python

            >>> from optilog.encoders.pb import IncrementalEncoder
            >>> IncrementalEncoder.card_encodings()
            ['best', 'bdd', 'card', 'totalizer']
        """
