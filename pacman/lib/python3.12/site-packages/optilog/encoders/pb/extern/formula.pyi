from _typeshed import Incomplete as Incomplete

aiger_present: bool

class IDPool:
    """
    A simple manager of variable IDs. It can be used as a pool of integers
    assigning an ID to any object. Identifiers are to start from ``1`` by
    default. The list of occupied intervals is empty be default. If
    necessary the top variable ID can be accessed directly using the
    ``top`` variable.

    :param start_from: the smallest ID to assign.
    :param occupied: a list of occupied intervals.

    :type start_from: int
    :type occupied: list(list(int))
    """
    def __init__(self, start_from: int = 1, occupied=[]) -> None:
        """
        Constructor.
        """
    top: Incomplete
    obj2id: Incomplete
    id2obj: Incomplete
    def restart(self, start_from: int = 1, occupied=[]):
        """
        Restart the manager from scratch. The arguments replicate those of
        the constructor of :class:`IDPool`.
        """
    def id(self, obj) -> None:
        """
        The method is to be used to assign an integer variable ID for a
        given new object. If the object already has an ID, no new ID is
        created and the old one is returned instead.

        An object can be anything. In some cases it is convenient to use
        string variable names.

        :param obj: an object to assign an ID to.

        :rtype: int.

        Example:

        .. code-block:: python

            >>> from pysat.formula import IDPool
            >>> vpool = IDPool(occupied=[[12, 18], [3, 10]])
            >>>
            >>> # creating 5 unique variables for the following strings
            >>> for i in range(5):
            ...    print(vpool.id('v{0}'.format(i + 1)))
            1
            2
            11
            19
            20

        In some cases, it makes sense to create an external function for
        accessing IDPool, e.g.:

        .. code-block:: python

            >>> # continuing the previous example
            >>> var = lambda i: vpool.id('var{0}'.format(i))
            >>> var(5)
            20
            >>> var('hello_world!')
            21
        """
    def obj(self, vid) -> None:
        """
        The method can be used to map back a given variable identifier to
        the original object labeled by the identifier.

        :param vid: variable identifier.
        :type vid: int

        :return: an object corresponding to the given identifier.

        Example:

        .. code-block:: python

            >>> vpool.obj(21)
            'hello_world!'
        """
    def occupy(self, start, stop) -> None:
        """
        Mark a given interval as occupied so that the manager could skip
        the values from ``start`` to ``stop`` (**inclusive**).

        :param start: beginning of the interval.
        :param stop: end of the interval.

        :type start: int
        :type stop: int
        """

class CNF:
    """
    Class for manipulating CNF formulas. It can be used for creating
    formulas, reading them from a file, or writing them to a file. The
    ``comment_lead`` parameter can be helpful when one needs to parse
    specific comment lines starting not with character ``c`` but with
    another character or a string.

    :param from_file: a DIMACS CNF filename to read from
    :param from_fp: a file pointer to read from
    :param from_string: a string storing a CNF formula
    :param from_clauses: a list of clauses to bootstrap the formula with
    :param from_aiger: an AIGER circuit to bootstrap the formula with
    :param comment_lead: a list of characters leading comment lines

    :type from_file: str
    :type from_fp: file_pointer
    :type from_string: str
    :type from_clauses: list(list(int))
    :type from_aiger: :class:`aiger.AIG` (see `py-aiger package <https://github.com/mvcisback/py-aiger>`__)
    :type comment_lead: list(str)
    """
    nv: int
    clauses: Incomplete
    comments: Incomplete
    def __init__(self, from_file: Incomplete | None = None, from_fp: Incomplete | None = None, from_string: Incomplete | None = None, from_clauses=[], from_aiger: Incomplete | None = None, comment_lead=['c']) -> None:
        """
        Constructor.
        """
    def from_file(self, fname, comment_lead=['c'], compressed_with: str = 'use_ext') -> None:
        """
        Read a CNF formula from a file in the DIMACS format. A file name is
        expected as an argument. A default argument is ``comment_lead`` for
        parsing comment lines. A given file can be compressed by either
        gzip, bzip2, or lzma.

        :param fname: name of a file to parse.
        :param comment_lead: a list of characters leading comment lines
        :param compressed_with: file compression algorithm

        :type fname: str
        :type comment_lead: list(str)
        :type compressed_with: str

        Note that the ``compressed_with`` parameter can be ``None`` (i.e.
        the file is uncompressed), ``'gzip'``, ``'bzip2'``, ``'lzma'``, or
        ``'use_ext'``. The latter value indicates that compression type
        should be automatically determined based on the file extension.
        Using ``'lzma'`` in Python 2 requires the ``backports.lzma``
        package to be additionally installed.

        Usage example:

        .. code-block:: python

            >>> from pysat.formula import CNF
            >>> cnf1 = CNF()
            >>> cnf1.from_file('some-file.cnf.gz', compressed_with='gzip')
            >>>
            >>> cnf2 = CNF(from_file='another-file.cnf')
        """
    def from_fp(self, file_pointer, comment_lead=['c']) -> None:
        """
        Read a CNF formula from a file pointer. A file pointer should be
        specified as an argument. The only default argument is
        ``comment_lead``, which can be used for parsing specific comment
        lines.

        :param file_pointer: a file pointer to read the formula from.
        :param comment_lead: a list of characters leading comment lines

        :type file_pointer: file pointer
        :type comment_lead: list(str)

        Usage example:

        .. code-block:: python

            >>> with open('some-file.cnf', 'r') as fp:
            ...     cnf1 = CNF()
            ...     cnf1.from_fp(fp)
            >>>
            >>> with open('another-file.cnf', 'r') as fp:
            ...     cnf2 = CNF(from_fp=fp)
        """
    def from_string(self, string, comment_lead=['c']) -> None:
        """
        Read a CNF formula from a string. The string should be specified as
        an argument and should be in the DIMACS CNF format. The only
        default argument is ``comment_lead``, which can be used for parsing
        specific comment lines.

        :param string: a string containing the formula in DIMACS.
        :param comment_lead: a list of characters leading comment lines

        :type string: str
        :type comment_lead: list(str)

        Example:

        .. code-block:: python

            >>> from pysat.formula import CNF
            >>> cnf1 = CNF()
            >>> cnf1.from_string(='p cnf 2 2\\n-1 2 0\\n1 -2 0')
            >>> print(cnf1.clauses)
            [[-1, 2], [1, -2]]
            >>>
            >>> cnf2 = CNF(from_string='p cnf 3 3\\n-1 2 0\\n-2 3 0\\n-3 0\\n')
            >>> print(cnf2.clauses)
            [[-1, 2], [-2, 3], [-3]]
            >>> print(cnf2.nv)
            3
        """
    def from_clauses(self, clauses) -> None:
        """
        This methods copies a list of clauses into a CNF object.

        :param clauses: a list of clauses
        :type clauses: list(list(int))

        Example:

        .. code-block:: python

            >>> from pysat.formula import CNF
            >>> cnf = CNF(from_clauses=[[-1, 2], [1, -2], [5]])
            >>> print(cnf.clauses)
            [[-1, 2], [1, -2], [5]]
            >>> print(cnf.nv)
            5
        """
    vpool: Incomplete
    inps: Incomplete
    outs: Incomplete
    def from_aiger(self, aig, vpool: Incomplete | None = None) -> None:
        """

        Create a CNF formula by Tseitin-encoding an input AIGER circuit.

        Input circuit is expected to be an object of class
        :class:`aiger.AIG`. Alternatively, it can be specified as an
        :class:`aiger.BoolExpr`, or an ``*.aag`` filename, or an AIGER
        string to parse. (Classes :class:`aiger.AIG` and
        :class:`aiger.BoolExpr` are defined in the `py-aiger package
        <https://github.com/mvcisback/py-aiger>`__.)

        :param aig: an input AIGER circuit
        :param vpool: pool of variable identifiers (optional)

        :type aig: :class:`aiger.AIG` (see `py-aiger package <https://github.com/mvcisback/py-aiger>`__)
        :type vpool: :class:`.IDPool`

        Example:

        .. code-block:: python

            >>> import aiger
            >>> x, y, z = aiger.atom('x'), aiger.atom('y'), aiger.atom('z')
            >>> expr = ~(x | y) & z
            >>> print(expr.aig)
            aag 5 3 0 1 2
            2
            4
            8
            10
            6 3 5
            10 6 8
            i0 y
            i1 x
            i2 z
            o0 6c454aea-c9e1-11e9-bbe3-3af9d34370a9
            >>>
            >>> from pysat.formula import CNF
            >>> cnf = CNF(from_aiger=expr.aig)
            >>> print(cnf.nv)
            5
            >>> print(cnf.clauses)
            [[3, 2, 4], [-3, -4], [-2, -4], [-4, -1, 5], [4, -5], [1, -5]]
            >>> print(['{0} <-> {1}'.format(v, cnf.vpool.obj(v)) for v in cnf.inps])
            ['3 <-> y', '2 <-> x', '1 <-> z']
            >>> print(['{0} <-> {1}'.format(v, cnf.vpool.obj(v)) for v in cnf.outs])
            ['5 <-> 6c454aea-c9e1-11e9-bbe3-3af9d34370a9']
        """
    def copy(self) -> None:
        """
        This method can be used for creating a copy of a CNF object. It
        creates another object of the :class:`CNF` class and makes use of
        the *deepcopy* functionality to copy the clauses.

        :return: an object of class :class:`CNF`.

        Example:

        .. code-block:: python

            >>> cnf1 = CNF(from_clauses=[[-1, 2], [1]])
            >>> cnf2 = cnf1.copy()
            >>> print(cnf2.clauses)
            [[-1, 2], [1]]
            >>> print(cnf2.nv)
            2
        """
    def to_file(self, fname, comments: Incomplete | None = None, compress_with: str = 'use_ext') -> None:
        """
        The method is for saving a CNF formula into a file in the DIMACS
        CNF format. A file name is expected as an argument. Additionally,
        supplementary comment lines can be specified in the ``comments``
        parameter. Also, a file can be compressed using either gzip, bzip2,
        or lzma (xz).

        :param fname: a file name where to store the formula.
        :param comments: additional comments to put in the file.
        :param compress_with: file compression algorithm

        :type fname: str
        :type comments: list(str)
        :type compress_with: str

        Note that the ``compress_with`` parameter can be ``None`` (i.e.
        the file is uncompressed), ``'gzip'``, ``'bzip2'``, ``'lzma'``, or
        ``'use_ext'``. The latter value indicates that compression type
        should be automatically determined based on the file extension.
        Using ``'lzma'`` in Python 2 requires the ``backports.lzma``
        package to be additionally installed.

        Example:

        .. code-block:: python

            >>> from pysat.formula import CNF
            >>> cnf = CNF()
            ...
            >>> # the formula is filled with a bunch of clauses
            >>> cnf.to_file('some-file-name.cnf')  # writing to a file
        """
    def to_fp(self, file_pointer, comments: Incomplete | None = None) -> None:
        """
        The method can be used to save a CNF formula into a file pointer.
        The file pointer is expected as an argument. Additionally,
        supplementary comment lines can be specified in the ``comments``
        parameter.

        :param fname: a file name where to store the formula.
        :param comments: additional comments to put in the file.

        :type fname: str
        :type comments: list(str)

        Example:

        .. code-block:: python

            >>> from pysat.formula import CNF
            >>> cnf = CNF()
            ...
            >>> # the formula is filled with a bunch of clauses
            >>> with open('some-file.cnf', 'w') as fp:
            ...     cnf.to_fp(fp)  # writing to the file pointer
        """
    def append(self, clause) -> None:
        """
        Add one more clause to CNF formula. This method additionally
        updates the number of variables, i.e. variable ``self.nv``, used in
        the formula.

        :param clause: a new clause to add.
        :type clause: list(int)

        .. code-block:: python

            >>> from pysat.formula import CNF
            >>> cnf = CNF(from_clauses=[[-1, 2], [3]])
            >>> cnf.append([-3, 4])
            >>> print(cnf.clauses)
            [[-1, 2], [3], [-3, 4]]
        """
    def extend(self, clauses) -> None:
        """
        Add several clauses to CNF formula. The clauses should be given in
        the form of list. For every clause in the list, method
        :meth:`append` is invoked.

        :param clauses: a list of new clauses to add.
        :type clauses: list(list(int))

        Example:

        .. code-block:: python

            >>> from pysat.formula import CNF
            >>> cnf = CNF(from_clauses=[[-1, 2], [3]])
            >>> cnf.extend([[-3, 4], [5, 6]])
            >>> print(cnf.clauses)
            [[-1, 2], [3], [-3, 4], [5, 6]]
        """
    def __iter__(self):
        """
        Iterator over all clauses of the formula.
        """
    def weighted(self) -> None:
        """
        This method creates a weighted copy of the internal formula. As a
        result, an object of class :class:`WCNF` is returned. Every clause
        of the CNF formula is *soft* in the new WCNF formula and its weight
        is equal to ``1``. The set of hard clauses of the formula is empty.

        :return: an object of class :class:`WCNF`.

        Example:

        .. code-block:: python

            >>> from pysat.formula import CNF
            >>> cnf = CNF(from_clauses=[[-1, 2], [3, 4]])
            >>>
            >>> wcnf = cnf.weighted()
            >>> print(wcnf.hard)
            []
            >>> print(wcnf.soft)
            [[-1, 2], [3, 4]]
            >>> print(wcnf.wght)
            [1, 1]
        """
    def negate(self, topv: Incomplete | None = None):
        """
        Given a CNF formula :math:`\\mathcal{F}`, this method creates a CNF
        formula :math:`\\neg{\\mathcal{F}}`. The negation of the formula is
        encoded to CNF with the use of *auxiliary* Tseitin variables [1]_.
        A new CNF formula is returned keeping all the newly introduced
        variables that can be accessed through the ``auxvars`` variable.

        **Note** that the negation of each clause is encoded with one
        auxiliary variable if it is not unit size. Otherwise, no auxiliary
        variable is introduced.

        :param topv: top variable identifier if any.
        :type topv: int

        :return: an object of class :class:`CNF`.

        .. [1] G. S. Tseitin. *On the complexity of derivations in the
            propositional calculus*.  Studies in Mathematics and
            Mathematical Logic, Part II. pp.  115–125, 1968

        .. code-block:: python

            >>> from pysat.formula import CNF
            >>> pos = CNF(from_clauses=[[-1, 2], [3]])
            >>> neg = pos.negate()
            >>> print(neg.clauses)
            [[1, -4], [-2, -4], [-1, 2, 4], [4, -3]]
            >>> print(neg.auxvars)
            [4, -3]
        """

class WCNF:
    """
    Class for manipulating partial (weighted) CNF formulas. It can be used
    for creating formulas, reading them from a file, or writing them to a
    file. The ``comment_lead`` parameter can be helpful when one needs to
    parse specific comment lines starting not with character ``c`` but with
    another character or a string.

    :param from_file: a DIMACS CNF filename to read from
    :param from_fp: a file pointer to read from
    :param from_string: a string storing a CNF formula
    :param comment_lead: a list of characters leading comment lines

    :type from_file: str
    :type from_fp: file_pointer
    :type from_string: str
    :type comment_lead: list(str)
    """
    nv: int
    hard: Incomplete
    soft: Incomplete
    wght: Incomplete
    topw: int
    comments: Incomplete
    def __init__(self, from_file: Incomplete | None = None, from_fp: Incomplete | None = None, from_string: Incomplete | None = None, comment_lead=['c']) -> None:
        """
        Constructor.
        """
    def from_file(self, fname, comment_lead=['c'], compressed_with: str = 'use_ext') -> None:
        """
        Read a WCNF formula from a file in the DIMACS format. A file name
        is expected as an argument. A default argument is ``comment_lead``
        for parsing comment lines. A given file can be compressed by either
        gzip, bzip2, or lzma.

        :param fname: name of a file to parse.
        :param comment_lead: a list of characters leading comment lines
        :param compressed_with: file compression algorithm

        :type fname: str
        :type comment_lead: list(str)
        :type compressed_with: str

        Note that the ``compressed_with`` parameter can be ``None`` (i.e.
        the file is uncompressed), ``'gzip'``, ``'bzip2'``, ``'lzma'``, or
        ``'use_ext'``. The latter value indicates that compression type
        should be automatically determined based on the file extension.
        Using ``'lzma'`` in Python 2 requires the ``backports.lzma``
        package to be additionally installed.

        Usage example:

        .. code-block:: python

            >>> from pysat.formula import WCNF
            >>> cnf1 = WCNF()
            >>> cnf1.from_file('some-file.wcnf.bz2', compressed_with='bzip2')
            >>>
            >>> cnf2 = WCNF(from_file='another-file.wcnf')
        """
    def from_fp(self, file_pointer, comment_lead=['c']) -> None:
        """
        Read a WCNF formula from a file pointer. A file pointer should be
        specified as an argument. The only default argument is
        ``comment_lead``, which can be used for parsing specific comment
        lines.

        :param file_pointer: a file pointer to read the formula from.
        :param comment_lead: a list of characters leading comment lines

        :type file_pointer: file pointer
        :type comment_lead: list(str)

        Usage example:

        .. code-block:: python

            >>> with open('some-file.cnf', 'r') as fp:
            ...     cnf1 = WCNF()
            ...     cnf1.from_fp(fp)
            >>>
            >>> with open('another-file.cnf', 'r') as fp:
            ...     cnf2 = WCNF(from_fp=fp)
        """
    def from_string(self, string, comment_lead=['c']) -> None:
        """
        Read a WCNF formula from a string. The string should be specified
        as an argument and should be in the DIMACS CNF format. The only
        default argument is ``comment_lead``, which can be used for parsing
        specific comment lines.

        :param string: a string containing the formula in DIMACS.
        :param comment_lead: a list of characters leading comment lines

        :type string: str
        :type comment_lead: list(str)

        Example:

        .. code-block:: python

            >>> from pysat.formula import WCNF
            >>> cnf1 = WCNF()
            >>> cnf1.from_string(='p wcnf 2 2 2\\n 2 -1 2 0\\n1 1 -2 0')
            >>> print(cnf1.hard)
            [[-1, 2]]
            >>> print(cnf1.soft)
            [[1, 2]]
            >>>
            >>> cnf2 = WCNF(from_string='p wcnf 3 3 2\\n2 -1 2 0\\n2 -2 3 0\\n1 -3 0\\n')
            >>> print(cnf2.hard)
            [[-1, 2], [-2, 3]]
            >>> print(cnf2.soft)
            [[-3]]
            >>> print(cnf2.nv)
            3
        """
    def copy(self) -> None:
        """
        This method can be used for creating a copy of a WCNF object. It
        creates another object of the :class:`WCNF` class and makes use of
        the *deepcopy* functionality to copy both hard and soft clauses.

        :return: an object of class :class:`WCNF`.

        Example:

        .. code-block:: python

            >>> cnf1 = WCNF()
            >>> cnf1.append([-1, 2])
            >>> cnf1.append([1], weight=10)
            >>>
            >>> cnf2 = cnf1.copy()
            >>> print(cnf2.hard)
            [[-1, 2]]
            >>> print(cnf2.soft)
            [[1]]
            >>> print(cnf2.wght)
            [10]
            >>> print(cnf2.nv)
            2
        """
    def to_file(self, fname, comments: Incomplete | None = None, compress_with: str = 'use_ext') -> None:
        """
        The method is for saving a WCNF formula into a file in the DIMACS
        CNF format. A file name is expected as an argument. Additionally,
        supplementary comment lines can be specified in the ``comments``
        parameter. Also, a file can be compressed using either gzip, bzip2,
        or lzma (xz).

        :param fname: a file name where to store the formula.
        :param comments: additional comments to put in the file.
        :param compress_with: file compression algorithm

        :type fname: str
        :type comments: list(str)
        :type compress_with: str

        Note that the ``compress_with`` parameter can be ``None`` (i.e.
        the file is uncompressed), ``'gzip'``, ``'bzip2'``, ``'lzma'``, or
        ``'use_ext'``. The latter value indicates that compression type
        should be automatically determined based on the file extension.
        Using ``'lzma'`` in Python 2 requires the ``backports.lzma``
        package to be additionally installed.

        Example:

        .. code-block:: python

            >>> from pysat.formula import WCNF
            >>> wcnf = WCNF()
            ...
            >>> # the formula is filled with a bunch of clauses
            >>> wcnf.to_file('some-file-name.wcnf')  # writing to a file
        """
    def to_fp(self, file_pointer, comments: Incomplete | None = None) -> None:
        """
        The method can be used to save a WCNF formula into a file pointer.
        The file pointer is expected as an argument. Additionally,
        supplementary comment lines can be specified in the ``comments``
        parameter.

        :param fname: a file name where to store the formula.
        :param comments: additional comments to put in the file.

        :type fname: str
        :type comments: list(str)

        Example:

        .. code-block:: python

            >>> from pysat.formula import WCNF
            >>> wcnf = WCNF()
            ...
            >>> # the formula is filled with a bunch of clauses
            >>> with open('some-file.wcnf', 'w') as fp:
            ...     wcnf.to_fp(fp)  # writing to the file pointer
        """
    def append(self, clause, weight: Incomplete | None = None) -> None:
        """
        Add one more clause to WCNF formula. This method additionally
        updates the number of variables, i.e. variable ``self.nv``, used in
        the formula.

        The clause can be hard or soft depending on the ``weight``
        argument. If no weight is set, the clause is considered to be hard.

        :param clause: a new clause to add.
        :param weight: integer weight of the clause.

        :type clause: list(int)
        :type weight: integer or None

        .. code-block:: python

            >>> from pysat.formula import WCNF
            >>> cnf = WCNF()
            >>> cnf.append([-1, 2])
            >>> cnf.append([1], weight=10)
            >>> cnf.append([-2], weight=20)
            >>> print(cnf.hard)
            [[-1, 2]]
            >>> print(cnf.soft)
            [[1], [-2]]
            >>> print(cnf.wght)
            [10, 20]
        """
    def extend(self, clauses, weights: Incomplete | None = None) -> None:
        """
        Add several clauses to WCNF formula. The clauses should be given in
        the form of list. For every clause in the list, method
        :meth:`append` is invoked.

        The clauses can be hard or soft depending on the ``weights``
        argument. If no weights are set, the clauses are considered to be
        hard.

        :param clauses: a list of new clauses to add.
        :param weights: a list of integer weights.

        :type clauses: list(list(int))
        :type weights: list(int)

        Example:

        .. code-block:: python

            >>> from pysat.formula import WCNF
            >>> cnf = WCNF()
            >>> cnf.extend([[-3, 4], [5, 6]])
            >>> cnf.extend([[3], [-4], [-5], [-6]], weights=[1, 5, 3, 4])
            >>> print(cnf.hard)
            [[-3, 4], [5, 6]]
            >>> print(cnf.soft)
            [[3], [-4], [-5], [-6]]
            >>> print(cnf.wght)
            [1, 5, 3, 4]
        """
    def unweighted(self) -> None:
        """
        This method creates a *plain* (unweighted) copy of the internal
        formula. As a result, an object of class :class:`CNF` is returned.
        Every clause (both hard or soft) of the WCNF formula is copied to
        the ``clauses`` variable of the resulting plain formula, i.e. all
        weights are discarded.

        :return: an object of class :class:`CNF`.

        Example:

        .. code-block:: python

            >>> from pysat.formula import WCNF
            >>> wcnf = WCNF()
            >>> wcnf.extend([[-3, 4], [5, 6]])
            >>> wcnf.extend([[3], [-4], [-5], [-6]], weights=[1, 5, 3, 4])
            >>>
            >>> cnf = wcnf.unweighted()
            >>> print(cnf.clauses)
            [[-3, 4], [5, 6], [3], [-4], [-5], [-6]]
        """

class CNFPlus(CNF):
    """
    CNF formulas augmented with *native* cardinality constraints.

    This class inherits most of the functionality of the :class:`CNF`
    class. The only difference between the two is that :class:`CNFPlus`
    supports *native* cardinality constraints of `MiniCard
    <https://github.com/liffiton/minicard>`__.

    The parser of input DIMACS files of :class:`CNFPlus` assumes the syntax
    of AtMostK and AtLeastK constraints defined in the `description
    <https://github.com/liffiton/minicard>`__ of MiniCard:

    ::

        c Example: Two cardinality constraints followed by a clause
        p cnf+ 7 3
        1 -2 3 5 -7 <= 3
        4 5 6 -7 >= 2
        3 5 7 0

    Each AtLeastK constraint is translated into an AtMostK constraint in
    the standard way: :math:`\\sum_{i=1}^{n}{x_i}\\geq k \\leftrightarrow
    \\sum_{i=1}^{n}{\\neg{x_i}}\\leq (n-k)`. Internally, AtMostK constraints
    are stored in variable ``atmosts``, each being a pair ``(lits, k)``,
    where ``lits`` is a list of literals in the sum and ``k`` is the upper
    bound.

    Example:

    .. code-block:: python

        >>> from pysat.formula import CNFPlus
        >>> cnf = CNFPlus(from_string='p cnf+ 7 3\\n1 -2 3 5 -7 <= 3\\n4 5 6 -7 >= 2\\n 3 5 7 0\\n')
        >>> print(cnf.clauses)
        [[3, 5, 7]]
        >>> print(cnf.atmosts)
        [[[1, -2, 3, 5, -7], 3], [[-4, -5, -6, 7], 2]]
        >>> print(cnf.nv)
        7

    For details on the functionality, see :class:`CNF`.
    """
    atmosts: Incomplete
    def __init__(self, from_file: Incomplete | None = None, from_fp: Incomplete | None = None, from_string: Incomplete | None = None, comment_lead=['c']) -> None:
        """
        Constructor.
        """
    nv: int
    clauses: Incomplete
    comments: Incomplete
    def from_fp(self, file_pointer, comment_lead=['c']) -> None:
        """
        Read a CNF+ formula from a file pointer. A file pointer should be
        specified as an argument. The only default argument is
        ``comment_lead``, which can be used for parsing specific comment
        lines.

        :param file_pointer: a file pointer to read the formula from.
        :param comment_lead: a list of characters leading comment lines

        :type file_pointer: file pointer
        :type comment_lead: list(str)

        Usage example:

        .. code-block:: python

            >>> with open('some-file.cnf+', 'r') as fp:
            ...     cnf1 = CNFPlus()
            ...     cnf1.from_fp(fp)
            >>>
            >>> with open('another-file.cnf+', 'r') as fp:
            ...     cnf2 = CNFPlus(from_fp=fp)
        """
    def to_fp(self, file_pointer, comments: Incomplete | None = None) -> None:
        """
        The method can be used to save a CNF+ formula into a file pointer.
        The file pointer is expected as an argument. Additionally,
        supplementary comment lines can be specified in the ``comments``
        parameter.

        :param fname: a file name where to store the formula.
        :param comments: additional comments to put in the file.

        :type fname: str
        :type comments: list(str)

        Example:

        .. code-block:: python

            >>> from pysat.formula import CNFPlus
            >>> cnf = CNFPlus()
            ...
            >>> # the formula is filled with a bunch of clauses
            >>> with open('some-file.cnf+', 'w') as fp:
            ...     cnf.to_fp(fp)  # writing to the file pointer
        """
    def append(self, clause, is_atmost: bool = False) -> None:
        """
        Add a single clause or a single AtMostK constraint to CNF+ formula.
        This method additionally updates the number of variables, i.e.
        variable ``self.nv``, used in the formula.

        If the clause is an AtMostK constraint, this should be set with the
        use of the additional default argument ``is_atmost``, which is set
        to ``False`` by default.

        :param clause: a new clause to add.
        :param is_atmost: if ``True``, the clause is AtMostK.

        :type clause: list(int)
        :type is_atmost: bool

        .. code-block:: python

            >>> from pysat.formula import CNFPlus
            >>> cnf = CNFPlus()
            >>> cnf.append([-3, 4])
            >>> cnf.append([[1, 2, 3], 1], is_atmost=True)
            >>> print(cnf.clauses)
            [[-3, 4]]
            >>> print(cnf.atmosts)
            [[1, 2, 3], 1]
        """
    def weighted(self) -> None:
        """
        This method creates a weighted copy of the internal formula. As a
        result, an object of class :class:`WCNFPlus` is returned. Every
        clause of the CNFPlus formula is *soft* in the new WCNFPlus
        formula and its weight is equal to ``1``. The set of hard clauses
        of the new formula is empty. The set of cardinality constraints
        remains unchanged.

        :return: an object of class :class:`WCNFPlus`.

        Example:

        .. code-block:: python

            >>> from pysat.formula import CNFPlus
            >>> cnf = CNFPlus()
            >>> cnf.append([-1, 2])
            >>> cnf.append([3, 4])
            >>> cnf.append([[1, 2], 1], is_atmost=True)
            >>>
            >>> wcnf = cnf.weighted()
            >>> print(wcnf.hard)
            []
            >>> print(wcnf.soft)
            [[-1, 2], [3, 4]]
            >>> print(wcnf.wght)
            [1, 1]
            >>> print(wcnf.atms)
            [[[1, 2], 1]]
        """
    def copy(self) -> None:
        """
        This method can be used for creating a copy of a CNFPlus object.
        It creates another object of the :class:`CNFPlus` class, call the
        copy function of CNF class and makes use of the *deepcopy*
        functionality to copy the atmost constraints.

        :return: an object of class :class:`CNFPlus`.

        Example:

        .. code-block:: python

            >>> cnf1 = CNFPlus()
            >>> cnf1.extend([[-1, 2], [1]])
            >>> cnf1.append([[1, 2], 1], is_atmost=True)
            >>> cnf2 = cnf1.copy()
            >>> print(cnf2.clauses)
            [[-1, 2], [1]]
            >>> print(cnf2.nv)
            2
            >>> print(cnf2.atmosts)
            [[[1, 2], 1]]
        """

class WCNFPlus(WCNF):
    """
    WCNF formulas augmented with *native* cardinality constraints.

    This class inherits most of the functionality of the :class:`WCNF`
    class. The only difference between the two is that :class:`WCNFPlus`
    supports *native* cardinality constraints of `MiniCard
    <https://github.com/liffiton/minicard>`__.

    The parser of input DIMACS files of :class:`WCNFPlus` assumes the
    syntax of AtMostK and AtLeastK constraints following the one defined
    for :class:`CNFPlus` in the `description
    <https://github.com/liffiton/minicard>`__ of MiniCard:

    ::

        c Example: Two (hard) cardinality constraints followed by a soft clause
        p wcnf+ 7 3 10
        10 1 -2 3 5 -7 <= 3
        10 4 5 6 -7 >= 2
        5 3 5 7 0

    **Note** that every cardinality constraint is assumed to be *hard*,
    i.e. soft cardinality constraints are currently *not supported*.

    Each AtLeastK constraint is translated into an AtMostK constraint in
    the standard way: :math:`\\sum_{i=1}^{n}{x_i}\\geq k \\leftrightarrow
    \\sum_{i=1}^{n}{\\neg{x_i}}\\leq (n-k)`. Internally, AtMostK constraints
    are stored in variable ``atms``, each being a pair ``(lits, k)``, where
    ``lits`` is a list of literals in the sum and ``k`` is the upper bound.

    Example:

    .. code-block:: python

        >>> from pysat.formula import WCNFPlus
        >>> cnf = WCNFPlus(from_string='p wcnf+ 7 3 10\\n10 1 -2 3 5 -7 <= 3\\n10 4 5 6 -7 >= 2\\n5 3 5 7 0\\n')
        >>> print(cnf.soft)
        [[3, 5, 7]]
        >>> print(cnf.wght)
        [5]
        >>> print(cnf.hard)
        []
        >>> print(cnf.atms)
        [[[1, -2, 3, 5, -7], 3], [[-4, -5, -6, 7], 2]]
        >>> print(cnf.nv)
        7

    For details on the functionality, see :class:`WCNF`.
    """
    atms: Incomplete
    def __init__(self, from_file: Incomplete | None = None, from_fp: Incomplete | None = None, from_string: Incomplete | None = None, comment_lead=['c']) -> None:
        """
        Constructor.
        """
    nv: int
    hard: Incomplete
    soft: Incomplete
    wght: Incomplete
    topw: int
    comments: Incomplete
    def from_fp(self, file_pointer, comment_lead=['c']) -> None:
        """
        Read a WCNF+ formula from a file pointer. A file pointer should be
        specified as an argument. The only default argument is
        ``comment_lead``, which can be used for parsing specific comment
        lines.

        :param file_pointer: a file pointer to read the formula from.
        :param comment_lead: a list of characters leading comment lines

        :type file_pointer: file pointer
        :type comment_lead: list(str)

        Usage example:

        .. code-block:: python

            >>> with open('some-file.wcnf+', 'r') as fp:
            ...     cnf1 = WCNFPlus()
            ...     cnf1.from_fp(fp)
            >>>
            >>> with open('another-file.wcnf+', 'r') as fp:
            ...     cnf2 = WCNFPlus(from_fp=fp)
        """
    def to_fp(self, file_pointer, comments: Incomplete | None = None) -> None:
        """
        The method can be used to save a WCNF+ formula into a file pointer.
        The file pointer is expected as an argument. Additionally,
        supplementary comment lines can be specified in the ``comments``
        parameter.

        :param fname: a file name where to store the formula.
        :param comments: additional comments to put in the file.

        :type fname: str
        :type comments: list(str)

        Example:

        .. code-block:: python

            >>> from pysat.formula import WCNFPlus
            >>> cnf = WCNFPlus()
            ...
            >>> # the formula is filled with a bunch of clauses
            >>> with open('some-file.wcnf+', 'w') as fp:
            ...     cnf.to_fp(fp)  # writing to the file pointer
        """
    def append(self, clause, weight: Incomplete | None = None, is_atmost: bool = False) -> None:
        """
        Add a single clause or a single AtMostK constraint to WCNF+
        formula. This method additionally updates the number of variables,
        i.e.  variable ``self.nv``, used in the formula.

        If the clause is an AtMostK constraint, this should be set with the
        use of the additional default argument ``is_atmost``, which is set
        to ``False`` by default.

        If ``is_atmost`` is set to ``False``, the clause can be either hard
        or soft depending on the ``weight`` argument. If no weight is
        specified, the clause is considered hard. Otherwise, the clause is
        soft.

        :param clause: a new clause to add.
        :param weight: an integer weight of the clause.
        :param is_atmost: if ``True``, the clause is AtMostK.

        :type clause: list(int)
        :type weight: integer or None
        :type is_atmost: bool

        .. code-block:: python

            >>> from pysat.formula import WCNFPlus
            >>> cnf = WCNFPlus()
            >>> cnf.append([-3, 4])
            >>> cnf.append([[1, 2, 3], 1], is_atmost=True)
            >>> cnf.append([-1, -2], weight=35)
            >>> print(cnf.hard)
            [[-3, 4]]
            >>> print(cnf.atms)
            [[1, 2, 3], 1]
            >>> print(cnf.soft)
            [[-1, -2]]
            >>> print(cnf.wght)
            [35]
        """
    def unweighted(self) -> None:
        """
        This method creates a *plain* (unweighted) copy of the internal
        formula. As a result, an object of class :class:`CNFPlus` is
        returned. Every clause (both hard or soft) of the original
        WCNFPlus formula is copied to the ``clauses`` variable of the
        resulting plain formula, i.e. all weights are discarded.

        Note that the cardinality constraints of the original (weighted)
        formula remain unchanged in the new (plain) formula.

        :return: an object of class :class:`CNFPlus`.

        Example:

        .. code-block:: python

            >>> from pysat.formula import WCNF
            >>> wcnf = WCNFPlus()
            >>> wcnf.extend([[-3, 4], [5, 6]])
            >>> wcnf.extend([[3], [-4], [-5], [-6]], weights=[1, 5, 3, 4])
            >>> wcnf.append([[1, 2, 3], 1], is_atmost=True)
            >>>
            >>> cnf = wcnf.unweighted()
            >>> print(cnf.clauses)
            [[-3, 4], [5, 6], [3], [-4], [-5], [-6]]
            >>> print(cnf.atmosts)
            [[[1, 2, 3], 1]]
        """
    def copy(self) -> None:
        """
        This method can be used for creating a copy of a WCNFPlus object.
        It creates another object of the :class:`WCNFPlus` class, call the
        copy function of WCNF class and makes use of the *deepcopy*
        functionality to copy the atmost constraints.

        :return: an object of class :class:`WCNFPlus`.

        Example:

        .. code-block:: python

            >>> cnf1 = WCNFPlus()
            >>> cnf1.append([-1, 2])
            >>> cnf1.append([1], weight=10)
            >>> cnf1.append([[1, 2], 1], is_atmost=True)
            >>> cnf2 = cnf1.copy()
            >>> print(cnf2.hard)
            [[-1, 2]]
            >>> print(cnf2.soft)
            [[1]]
            >>> print(cnf2.wght)
            [10]
            >>> print(cnf2.nv)
            2
            >> print(cnf2.atms)
            [[[1, 2], 1]]

        """
