from _typeshed import Incomplete as Incomplete

class NoSuchEncodingError(Exception):
    """
    This exception is raised when creating an unknown an AtMostk, AtLeastK,
    or EqualK constraint encoding.
    """

class EncType:
    '''
    This class represents a C-like ``enum`` type for choosing the
    cardinality encoding to use. The values denoting the encodings are:

    ::

        pairwise    = 0
        seqcounter  = 1
        sortnetwrk  = 2
        cardnetwrk  = 3
        bitwise     = 4
        ladder      = 5
        totalizer   = 6
        mtotalizer  = 7
        kmtotalizer = 8
        native      = 9

    The desired encoding can be selected either directly by its integer
    identifier, e.g. ``2``, or by its alphabetical name, e.g.
    ``EncType.sortnetwrk``.

    Note that while most of the encodings are produced as a list of
    clauses, the "native" encoding of `MiniCard
    <https://github.com/liffiton/minicard>`__ is managed as one clause.
    Given an AtMostK constraint :math:`\\sum_{i=1}^n{x_i\\leq k}`, the native
    encoding represents it as a pair ``[lits, k]``, where ``lits`` is a
    list of size ``n`` containing literals in the sum.
    '''
    pairwise: int
    seqcounter: int
    sortnetwrk: int
    cardnetwrk: int
    bitwise: int
    ladder: int
    totalizer: int
    mtotalizer: int
    kmtotalizer: int
    native: int

class CardEnc:
    """
    This abstract class is responsible for the creation of cardinality
    constraints encoded to a CNF formula. The class has three *class
    methods* for creating AtMostK, AtLeastK, and EqualsK constraints. Given
    a list of literals, an integer bound and an encoding type, each of
    these methods returns an object of class :class:`pysat.formula.CNFPlus`
    representing the resulting CNF formula.

    Since the class is abstract, there is no need to create an object of
    it. Instead, the methods should be called directly as class methods,
    e.g. ``CardEnc.atmost(lits, bound)`` or ``CardEnc.equals(lits,
    bound)``. An example usage is the following:

    .. code-block:: python

        >>> from pysat.card import *
        >>> cnf = CardEnc.atmost(lits=[1, 2, 3], encoding=EncType.pairwise)
        >>> print(cnf.clauses)
        [[-1, -2], [-1, -3], [-2, -3]]
        >>> cnf = CardEnc.equals(lits=[1, 2, 3], encoding=EncType.pairwise)
        >>> print(cnf.clauses)
        [[1, 2, 3], [-1, -2], [-1, -3], [-2, -3]]
    """
    @classmethod
    def atmost(cls, lits, bound: int = 1, top_id: Incomplete | None = None, vpool: Incomplete | None = None, encoding=...):
        """
        This method can be used for creating a CNF encoding of an AtMostK
        constraint, i.e. of :math:`\\sum_{i=1}^{n}{x_i}\\leq k`. The method
        shares the arguments and the return type with method
        :meth:`CardEnc.atleast`. Please, see it for details.
        """
    @classmethod
    def atleast(cls, lits, bound: int = 1, top_id: Incomplete | None = None, vpool: Incomplete | None = None, encoding=...):
        """
            This method can be used for creating a CNF encoding of an AtLeastK
            constraint, i.e. of :math:`\\sum_{i=1}^{n}{x_i}\\geq k`. The method
            takes 1 mandatory argument ``lits`` and 3 default arguments can be
            specified: ``bound``, ``top_id``, ``vpool``, and ``encoding``.

            :param lits: a list of literals in the sum.
            :param bound: the value of bound :math:`k`.
            :param top_id: top variable identifier used so far.
            :param vpool: variable pool for counting the number of variables.
            :param encoding: identifier of the encoding to use.

            :type lits: iterable(int)
            :type bound: int
            :type top_id: integer or None
            :type vpool: :class:`.IDPool`
            :type encoding: integer

            Parameter ``top_id`` serves to increase integer identifiers of
            auxiliary variables introduced during the encoding process. This
            is helpful when augmenting an existing CNF formula with the new
            cardinality encoding to make sure there is no collision between
            identifiers of the variables. If specified, the identifiers of the
            first auxiliary variable will be ``top_id+1``.

            Instead of ``top_id``, one may want to use a pool of variable
            identifiers ``vpool``, which is automatically updated during the
            method call. In many circumstances, this is more convenient than
            using ``top_id``. Also note that parameters ``top_id`` and
            ``vpool`` **cannot** be specified *simultaneusly*.

            The default value of ``encoding`` is :attr:`Enctype.seqcounter`.

            The method *translates* the AtLeast constraint into an AtMost
            constraint by *negating* the literals of ``lits``, creating a new
            bound :math:`n-k` and invoking :meth:`CardEnc.atmost` with the
            modified list of literals and the new bound.

            :raises CardEnc.NoSuchEncodingError: if encoding does not exist.

            :rtype: a :class:`.CNFPlus` object where the new             clauses (or the new native atmost constraint) are stored.
        """
    @classmethod
    def equals(cls, lits, bound: int = 1, top_id: Incomplete | None = None, vpool: Incomplete | None = None, encoding=...):
        """
        This method can be used for creating a CNF encoding of an EqualsK
        constraint, i.e. of :math:`\\sum_{i=1}^{n}{x_i}= k`. The method
        makes consecutive calls of both :meth:`CardEnc.atleast` and
        :meth:`CardEnc.atmost`. It shares the arguments and the return type
        with method :meth:`CardEnc.atleast`. Please, see it for details.
        """

class ITotalizer:
    """
    This class implements the iterative totalizer encoding [11]_. Note that
    :class:`ITotalizer` can be used only for creating AtMostK constraints.
    In contrast to class :class:`EncType`, this class is not abstract and
    its objects once created can be reused several times. The idea is that
    a *totalizer tree* can be extended, or the bound can be increased, as
    well as two totalizer trees can be merged into one.

    The constructor of the class object takes 3 default arguments.

    :param lits: a list of literals to sum.
    :param ubound: the largest potential bound to use.
    :param top_id: top variable identifier used so far.

    :type lits: iterable(int)
    :type ubound: int
    :type top_id: integer or None

    The encoding of the current tree can be accessed with the use of
    :class:`.CNF` variable stored as ``self.cnf``. Potential bounds **are
    not** imposed by default but can be added as unit clauses in the final
    CNF formula. The bounds are stored in the list of Boolean variables as
    ``self.rhs``. A concrete bound :math:`k` can be enforced by considering
    a unit clause ``-self.rhs[k]``. **Note** that ``-self.rhs[0]`` enforces
    all literals of the sum to be *false*.

    An :class:`ITotalizer` object should be deleted if it is not needed
    anymore.

    Possible usage of the class is shown below:

    .. code-block:: python

        >>> from pysat.card import ITotalizer
        >>> t = ITotalizer(lits=[1, 2, 3], ubound=1)
        >>> print(t.cnf.clauses)
        [[-2, 4], [-1, 4], [-1, -2, 5], [-4, 6], [-5, 7], [-3, 6], [-3, -4, 7]]
        >>> print(t.rhs)
        [6, 7]
        >>> t.delete()

    Alternatively, an object can be created using the ``with`` keyword. In
    this case, the object is deleted automatically:

    .. code-block:: python

        >>> from pysat.card import ITotalizer
        >>> with ITotalizer(lits=[1, 2, 3], ubound=1) as t:
        ...     print(t.cnf.clauses)
        [[-2, 4], [-1, 4], [-1, -2, 5], [-4, 6], [-5, 7], [-3, 6], [-3, -4, 7]]
        ...     print(t.rhs)
        [6, 7]
    """
    tobj: Incomplete
    lits: Incomplete
    ubound: int
    top_id: int
    cnf: Incomplete
    rhs: Incomplete
    nof_new: int
    def __init__(self, lits=[], ubound: int = 1, top_id: Incomplete | None = None) -> None:
        """
        Constructor.
        """
    def new(self, lits=[], ubound: int = 1, top_id: Incomplete | None = None):
        """
        The actual constructor of :class:`ITotalizer`. Invoked from
        ``self.__init__()``. Creates an object of :class:`ITotalizer` given
        a list of literals in the sum, the largest potential bound to
        consider, as well as the top variable identifier used so far. See
        the description of :class:`ITotalizer` for details.
        """
    def delete(self) -> None:
        """
        Destroys a previously constructed :class:`ITotalizer` object.
        Internal variables ``self.cnf`` and ``self.rhs`` get cleaned.
        """
    def __enter__(self) -> None:
        """
        'with' constructor.
        """
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: types.TracebackType | None) -> None:
        """
        'with' destructor.
        """
    def __del__(self) -> None:
        """
        Destructor.
        """
    def increase(self, ubound: int = 1, top_id: Incomplete | None = None) -> None:
        """
        Increases a potential upper bound that can be imposed on the
        literals in the sum of an existing :class:`ITotalizer` object to a
        new value.

        :param ubound: a new upper bound.
        :param top_id: a new top variable identifier.

        :type ubound: int
        :type top_id: integer or None

        The top identifier ``top_id`` applied only if it is greater than
        the one used in ``self``.

        This method creates additional clauses encoding the existing
        totalizer tree up to the new upper bound given and appends them to
        the list of clauses of :class:`.CNF` ``self.cnf``. The number of
        newly created clauses is stored in variable ``self.nof_new``.

        Also, a list of bounds ``self.rhs`` gets increased and its length
        becomes ``ubound+1``.

        The method can be used in the following way:

        .. code-block:: python

            >>> from pysat.card import ITotalizer
            >>> t = ITotalizer(lits=[1, 2, 3], ubound=1)
            >>> print(t.cnf.clauses)
            [[-2, 4], [-1, 4], [-1, -2, 5], [-4, 6], [-5, 7], [-3, 6], [-3, -4, 7]]
            >>> print(t.rhs)
            [6, 7]
            >>>
            >>> t.increase(ubound=2)
            >>> print(t.cnf.clauses)
            [[-2, 4], [-1, 4], [-1, -2, 5], [-4, 6], [-5, 7], [-3, 6], [-3, -4, 7], [-3, -5, 8]]
            >>> print(t.cnf.clauses[-t.nof_new:])
            [[-3, -5, 8]]
            >>> print(t.rhs)
            [6, 7, 8]
            >>> t.delete()
        """
    def extend(self, lits=[], ubound: Incomplete | None = None, top_id: Incomplete | None = None):
        """
        Extends the list of literals in the sum and (if needed) increases a
        potential upper bound that can be imposed on the complete list of
        literals in the sum of an existing :class:`ITotalizer` object to a
        new value.

        :param lits: additional literals to be included in the sum.
        :param ubound: a new upper bound.
        :param top_id: a new top variable identifier.

        :type lits: iterable(int)
        :type ubound: int
        :type top_id: integer or None

        The top identifier ``top_id`` applied only if it is greater than
        the one used in ``self``.

        This method creates additional clauses encoding the existing
        totalizer tree augmented with new literals in the sum and updating
        the upper bound. As a result, it appends the new clauses to the
        list of clauses of :class:`.CNF` ``self.cnf``. The number of newly
        created clauses is stored in variable ``self.nof_new``.

        Also, if the upper bound is updated, a list of bounds ``self.rhs``
        gets increased and its length becomes ``ubound+1``. Otherwise, it
        is updated with new values.

        The method can be used in the following way:

        .. code-block:: python

            >>> from pysat.card import ITotalizer
            >>> t = ITotalizer(lits=[1, 2], ubound=1)
            >>> print(t.cnf.clauses)
            [[-2, 3], [-1, 3], [-1, -2, 4]]
            >>> print(t.rhs)
            [3, 4]
            >>>
            >>> t.extend(lits=[5], ubound=2)
            >>> print(t.cnf.clauses)
            [[-2, 3], [-1, 3], [-1, -2, 4], [-5, 6], [-3, 6], [-4, 7], [-3, -5, 7], [-4, -5, 8]]
            >>> print(t.cnf.clauses[-t.nof_new:])
            [[-5, 6], [-3, 6], [-4, 7], [-3, -5, 7], [-4, -5, 8]]
            >>> print(t.rhs)
            [6, 7, 8]
            >>> t.delete()
        """
    def merge_with(self, another, ubound: Incomplete | None = None, top_id: Incomplete | None = None) -> None:
        """
        This method merges a tree of the current :class:`ITotalizer`
        object, with a tree of another object and (if needed) increases a
        potential upper bound that can be imposed on the complete list of
        literals in the sum of an existing :class:`ITotalizer` object to a
        new value.

        :param another: another totalizer to merge with.
        :param ubound: a new upper bound.
        :param top_id: a new top variable identifier.

        :type another: :class:`ITotalizer`
        :type ubound: int
        :type top_id: integer or None

        The top identifier ``top_id`` applied only if it is greater than
        the one used in ``self``.

        This method creates additional clauses encoding the existing
        totalizer tree merged with another totalizer tree into *one* sum
        and updating the upper bound. As a result, it appends the new
        clauses to the list of clauses of :class:`.CNF` ``self.cnf``. The
        number of newly created clauses is stored in variable
        ``self.nof_new``.

        Also, if the upper bound is updated, a list of bounds ``self.rhs``
        gets increased and its length becomes ``ubound+1``. Otherwise, it
        is updated with new values.

        The method can be used in the following way:

        .. code-block:: python

            >>> from pysat.card import ITotalizer
            >>> with ITotalizer(lits=[1, 2], ubound=1) as t1:
            ...     print(t1.cnf.clauses)
            [[-2, 3], [-1, 3], [-1, -2, 4]]
            ...     print(t1.rhs)
            [3, 4]
            ...
            ...     t2 = ITotalizer(lits=[5, 6], ubound=1)
            ...     print(t1.cnf.clauses)
            [[-6, 7], [-5, 7], [-5, -6, 8]]
            ...     print(t1.rhs)
            [7, 8]
            ...
            ...     t1.merge_with(t2)
            ...     print(t1.cnf.clauses)
            [[-2, 3], [-1, 3], [-1, -2, 4], [-6, 7], [-5, 7], [-5, -6, 8], [-7, 9], [-8, 10], [-3, 9], [-4, 10], [-3, -7, 10]]
            ...     print(t1.cnf.clauses[-t1.nof_new:])
            [[-6, 7], [-5, 7], [-5, -6, 8], [-7, 9], [-8, 10], [-3, 9], [-4, 10], [-3, -7, 10]]
            ...     print(t1.rhs)
            [9, 10]
            ...
            ...     t2.delete()
        """
