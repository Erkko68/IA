from optilog.formulas import CNF as CNF, WCNF as WCNF
from pathlib import Path

def load_cnf(file_path: str | Path, *args, **kwargs) -> CNF:
    """
    Loads a CNF DIMACS [#cnf]_ into a :py:class:`optilog.formulas.CNF` object and returns it.
    This method is particularly useful when working with big formulas
    because it is very performant.

    :param file_path: The path of a CNF in DIMACS format.

    .. code-block:: python

        >>> from optilog.formulas.loaders import load_cnf
        >>> cnf = load_cnf('example.cnf')
    """
def load_wcnf(file_path: str | Path, *args, **kwargs) -> WCNF:
    """
    Loads a WCNF DIMACS [#wcnf]_ into a :py:class:`optilog.formulas.WCNF` object and returns it.
    This method is particularly useful when working with big formulas
    because it is very performant.

    :param file_path: The path of a CNF in DIMACS format.

    .. code-block:: python

        >>> from optilog.formulas.loaders import load_wcnf
        >>> wcnf = load_wcnf('example.wcnf')

    """
