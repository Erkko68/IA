from optilog.formulas import QCNF as QCNF
from pathlib import Path

def load_qcnf(file_path: str | Path) -> QCNF:
    """Loads a QCNF DIMACS [#qcnf]_ into a :py:class:`optilog.formulas.QCNF` object and returns it.

    :param path: The path of a qcnf in DIMACS format.
    """
