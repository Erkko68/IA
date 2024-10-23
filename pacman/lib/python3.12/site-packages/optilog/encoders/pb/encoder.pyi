from .extern.card import ITotalizer as ITotalizer
from _typeshed import Incomplete as Incomplete

class PBIncrementalContext:
    incr_constraint: Incomplete
    aux_var: Incomplete
    config: Incomplete
    init_bound: Incomplete
    def __init__(self, incr_constraint, aux_var, config, bound) -> None: ...

class Encoder:
    """
    This class provides access to the most common Pseudo-Boolean and Cardinality constraint encodings
    implemented in PBLib.

    All the methods in this class are static.

    .. _constraint-types:

    List of available constraint types for the ``Encoder`` class:

    - At most one: :py:meth:`Encoder.at_most_one`
    - At least one: :py:meth:`Encoder.at_least_one`
    - Exactly one: :py:meth:`Encoder.exactly_one`
    - At most k: :py:meth:`Encoder.at_most_k`
    - At least k: :py:meth:`Encoder.at_least_k`
    - Exactly k: :py:meth:`Encoder.exactly_k`
    - Range k1 - k2: :py:meth:`Encoder.range_k1_k2`

    .. _pb-encodings:

    List of available Pseudo-Boolean encodings:

    - Adder networks :cite:p:`een2006translating`
    - Binary merge :cite:p:`manthey2014more`
    - BDD :cite:p:`een2006translating` :cite:p:`abio2011bdds`
    - Sequential weigh counters :cite:p:`holldobler2012compact`
    - Sorting networks :cite:p:`een2006translating`

    .. _card-encodings:

    List of available Cardinality encodings: 

    - bitwise :cite:p:`prestwich2009cnf`
    - cardinality network :cite:p:`asin2009cardinality`
    - ladder / regular :cite:p:`ansotegui2004mapping`
    - pairwise :cite:p:`prestwich2009cnf`
    - sequential counters :cite:p:`sinz2005towards`
    - sorting networks :cite:p:`asin2009cardinality`
    - totalizer :cite:p:`bailleux2003efficient`
    - totalizer modulo :cite:p:`ogawa2013modulo`
    - totalizer modulo for `k`-cardinality :cite:p:`morgado2014mscg`
    
    """
    cardenc: Incomplete
    pbenc: Incomplete
    inc_pbenc: Incomplete
    amoenc: Incomplete
    amkenc: Incomplete
    pbcomp: Incomplete
    card_at_k_method: Incomplete
    error_not_supported: Incomplete
    @staticmethod
    def pb_encodings() -> list[str]:
        """Returns a list of strings with all the available encodings for Pseudo-Boolean constraints.

        .. code-block:: python

            >>> from optilog.encoders.pb import Encoder
            >>> Encoder.pb_encodings()
            ['best', 'bdd', 'seqcounter', 'sortnetwork', 'adder', 'binarymerge']
        """
    @staticmethod
    def card_encodings() -> list[str]:
        """
        Returns a list of strings with all the available encodings for Cardinality constraints.

        .. code-block:: python

            >>> from optilog.encoders.pb import Encoder
            >>> Encoder.card_encodings()
            ['best', 'pairwise', 'seqcounter', 'sortnetwrk', 'cardnetwrk', 'bitwise', 'ladder', 'totalizer', 'mtotalizer', 'kmtotalizer']
        """
    @staticmethod
    def get_incremental_pb_encodings() -> list[str]:
        """
        Returns a list of strings with all the available incremental encodings for Pseudo-Boolean constraints.

        .. code-block:: python

            >>> from optilog.encoders.pb import Encoder
            >>> Encoder.get_incremental_pb_encodings()
            ['best', 'seqcounter', 'adder']
        """
    @staticmethod
    def get_incremental_amo_encodings() -> list[str]: ...
    @staticmethod
    def get_incremental_amk_encodings() -> list[str]: ...
    @staticmethod
    def get_max_var(literals: list[int]) -> int: ...
    @staticmethod
    def at_most_one(literals: list[int], weights: list[int] | None = None, max_var: int | None = None, encoding: str = 'best') -> tuple[int, list[list[int]]]:
        """
        Encodes the constraint :math:`\\sum_{i=1}^{n} w_i l_i \\leq 1` to CNF using the provided encoding. 

        :param literals: List of literals (:math:`l_i`).
        :param weights: List of weights (:math:`w_i`).
        :param max_var: Maximum variable used so far (default: max(literals)).
        :param encoding: Encoding (see :py:meth:`card_encodings` and :py:meth:`pb_encodings`).
        :return: Maximum DIMACS variable id used to encode the constraint and the list of SAT clauses encoding the constraint.

        :math:`x + 2y + z \\leq 1` (Pseudo-Boolean)

        .. code-block:: python

            >>> from optilog.encoders.pb import Encoder
            >>> Encoder.at_most_one(literals=[1,2,3], weights=[1,2,1])
            (3, [[-2], [-3, -1]])

        :math:`x + y + z \\leq 1` (Cardinality)

        .. code-block:: python

            >>> Encoder.at_most_one(literals=[1,2,3])
            (5, [[4, -1], [-2, 5], [-4, 5], [-2, -4], [-3, -5]])

        """
    @staticmethod
    def at_least_one(literals: list[int], weights: list[int] | None = None, max_var: int | None = None, encoding: str = 'best') -> tuple[int, list[list[int]]]:
        """
        Encodes the constraint :math:`\\sum_{i=1}^{n} w_i l_i \\geq 1` to CNF using the provided encoding.

        :param literals: List of literals (:math:`l_i`).
        :param weights: List of weights (:math:`w_i`).
        :param max_var: Maximum variable used so far (default: max(literals)).
        :param encoding: Encoding (see :py:meth:`card_encodings` and :py:meth:`pb_encodings`).
        :return: Maximum DIMACS variable id used to encode the constraint and the list of SAT clauses encoding the constraint.

        :math:`x + 2y + z \\geq 1` (Pseudo-Boolean)

        .. code-block:: python

            >>> from optilog.encoders.pb import Encoder
            >>> Encoder.at_least_one([1,2,3], weights=[1,2,1])
            (6, [[4], [1, 3, -5], [5, 2, -6], [6]])

        :math:`x + y + z \\geq 1` (Cardinality)

        .. code-block:: python

            >>> Encoder.at_least_one([1,2,3])
            (3, [[1, 2, 3]])
        """
    @staticmethod
    def exactly_one(literals: list[int], weights: list[int] | None = None, max_var: int | None = None, encoding: str = 'best') -> tuple[int, list[list[int]]]:
        """
        Encodes the constraint :math:`\\sum_{i=1}^{n} w_i l_i = 1` to CNF using the provided encoding.

        :param literals: List of literals (:math:`l_i`).
        :param weights: List of weights (:math:`w_i`).
        :param max_var: Maximum variable used so far (default: max(literals)).
        :param encoding: Encoding (see :py:meth:`card_encodings` and :py:meth:`pb_encodings`).
        :return: Maximum DIMACS variable id used to encode the constraint and the list of SAT clauses encoding the constraint.

        
        :math:`2x + y + z = 1` (Pseudo-Boolean)

        .. code-block:: python

            >>> from optilog.encoders.pb import Encoder
            >>> Encoder.exactly_one([1,2,3], weights=[2,1,1])
            (3, [[-1], [3, 2], [-3, -2]])

        :math:`x + y + z = 1` (Cardinality)
        
        .. code-block:: python

            >>> Encoder.exactly_one([1,2,3])
            (5, [[1, 2, 3], [4, -1], [-2, 5], [-4, 5], [-2, -4], [-3, -5]])

        """
    @staticmethod
    def at_most_k(literals: list[int], bound: int, weights: list[int] | None = None, max_var: int | None = None, encoding: str = 'best') -> tuple[int, list[list[int]]]:
        """
        Encodes the constraint :math:`\\sum_{i=1}^{n} w_i l_i \\leq k` to CNF using the provided encoding.

        :param literals: List of literals (:math:`l_i`).
        :param weights: List of weights (:math:`w_i`).
        :param bound: Upper bound of the constraint, using 1 is equivalent to call :py:meth:`at_most_one`
        :param max_var: Maximum variable used so far (default: max(literals)).
        :param encoding: Encoding (see :py:meth:`card_encodings` and :py:meth:`pb_encodings`).
        :return: Maximum DIMACS variable id used to encode the constraint and the list of SAT clauses encoding the constraint.

        
        :math:`2x + y + z \\leq 2` (Pseudo-Boolean)

        .. code-block:: python

            >>> from optilog.encoders.pb import Encoder
            >>> Encoder.at_most_k([1,2,3], 2, weights=[2,1,1])
            (9, [[4], [-4, 5], [-3, 5], [-4, -3, 6], [-2, 5], [-4, -2, 6], [-3, -2, 6], [-4, -3, -2, 7], [-1, 8], [-6, 8], [-1, -6, 9], [-9]])

        :math:`x + y + z \\leq 2` (Cardinality)

        .. code-block:: python

            >>> Encoder.at_most_k([1,2,3], 2)
            (3, [[-1, -2, -3]])

        """
    @staticmethod
    def at_least_k(literals: list[int], bound: int, weights: list[int] | None = None, max_var: int | None = None, encoding: str = 'best') -> tuple[int, list[list[int]]]:
        """
        Encodes the constraint :math:`\\sum_{i=1}^{n} w_i l_i \\geq k` to CNF using the provided encoding.

        :param literals: List of literals (:math:`l_i`).
        :param weights: List of weights (:math:`w_i`).
        :param bound: Upper bound of the constraint, using 1 is equivalent to call :py:meth:`at_most_one`
        :param max_var: Maximum variable used so far (default: max(literals)).
        :param encoding: Encoding (see :py:meth:`card_encodings` and :py:meth:`pb_encodings`).
        :return: Maximum DIMACS variable id used to encode the constraint and the list of SAT clauses encoding the constraint.

        :math:`2x + y + z \\geq 2` (Pseudo-Boolean)

        .. code-block:: python

            >>> from optilog.encoders.pb import Encoder
            >>> Encoder.at_least_k([1,2,3], 2, weights=[2,1,1])
            (9, [[4], [-4, 5], [3, 5], [-4, 3, 6], [2, 5], [-4, 2, 6], [3, 2, 6], [-4, 3, 2, 7], [1, 8], [-6, 8], [1, -6, 9], [-9]])

        :math:`x + y + z \\geq 2` (Cardinality)

        .. code-block:: python

            >>> Encoder.at_least_k([1,2,3], 2)
            (5, [[4, 1], [2, 5], [-4, 5], [2, -4], [3, -5]])

        """
    @staticmethod
    def exactly_k(literals: list[int], bound: int, weights: list[int] | None = None, max_var: int | None = None, encoding: str = 'best') -> tuple[int, list[list[int]]]:
        """
        Encodes the constraint :math:`\\sum_{i=1}^{n} w_i l_i = k` to CNF using the provided encoding.

        :param literals: List of literals (:math:`l_i`).
        :param weights: List of weights (:math:`w_i`).
        :param bound: Upper bound of the constraint, using 1 is equivalent to call :py:meth:`at_most_one`
        :param max_var: Maximum variable used so far (default: max(literals)).
        :param encoding: Encoding (see :py:meth:`card_encodings` and :py:meth:`pb_encodings`).
        :return: Maximum DIMACS variable id used to encode the constraint and the list of SAT clauses encoding the constraint.

        :math:`2x + y + z = 2` (Pseudo-Boolean)

        .. code-block:: python

            >>> from optilog.encoders.pb import Encoder
            >>> Encoder.exactly_k([1,2,3], 2, weights=[2,1,1])
            (8, [[4], [-5, -3], [-5, 3, -2], [-5, -2], [-6, 2], [-7, -3, 6], [-7, 3], [-7, 6], [-8, -1, 5], [-8, 1, 7], [-8, 5, 7], [8]])

        :math:`x + y + z = 2` (Cardinality)

        .. code-block:: python

            >>> Encoder.exactly_k([1,2,3], 2)
            (5, [[4, 1], [2, 5], [-4, 5], [2, -4], [3, -5], [-1, -2, -3]])


        """
    @staticmethod
    def range_k1_k2(literals: list[int], leqbound: int, geqbound: int, weights: list[int] | None = None, max_var: int | None = None, encoding: str = 'best') -> tuple[int, list[list[int]]]:
        """
        Encodes the constraint :math:`k2 \\leq \\sum_{i=1}^{n} w_i l_i \\leq k1` to CNF using the provided encoding.

        :param literals: List of literals (:math:`l_i`).
        :param weights: List of weights (:math:`w_i`).
        :param leqbound: The upper bound of the constraint, *k1*
        :param geqbound: The lower bound of the constraint, *k2*
        :param : Maximum variable used so far (default: max(literals)).
        :param encoding: Encoding (see :py:meth:`card_encodings` and :py:meth:`pb_encodings`).
        :return: Maximum DIMACS variable id used to encode the constraint and the list of SAT clauses encoding the constraint.

        :math:`2 \\leq 2x + y + z \\leq 3` (Pseudo-Boolean)

        .. code-block:: python

            >>> from optilog.encoders.pb import Encoder
            >>> Encoder.range_k1_k2([1,2,3], 3, 2, weights=[2,1,1])
            (9, [[-3, -2, -1], [4], [-4, 5], [3, 5], [-4, 3, 6], [2, 5], [-4, 2, 6], [3, 2, 6], [-4, 3, 2, 7], [1, 8], [-6, 8], [1, -6, 9], [-9]])

        :math:`1 \\leq x + y + z \\leq 2` (Cardinality)

        .. code-block:: python

            >>> Encoder.range_k1_k2([1,2,3], 2, 1)
            (3, [[1, 2, 3], [-1, -2, -3]])

        """
    @staticmethod
    def init_inc_at_most_k(literals, bound: int = 1, weights: Incomplete | None = None, max_var: Incomplete | None = None, encoding: str = 'best') -> tuple[int, list[list[int]], PBIncrementalContext]: ...
    @staticmethod
    def extend_inc_at_most_k(context: ITotalizer | PBIncrementalContext, newbound: int): ...
