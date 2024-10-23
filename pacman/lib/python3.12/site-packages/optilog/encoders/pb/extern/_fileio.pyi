from _typeshed import Incomplete as Incomplete

lzma_present: bool

class FileObject:
    """
    Auxiliary class for convenient and uniform file manipulation, e.g. to
    open files creating standard file pointers and closing them. The class
    is used when opening DIMACS files for reading and writing. Supports
    both uncompressed and compressed files. Compression algorithms
    supported are ``gzip``, ``bzip2``, and ``lzma``. Algorithm ``lzma`` can
    be used in Python 3 by default and also in Python 2 if the
    ``backports.lzma`` package is installed.

    Note that the class opens a file in text mode.

    :param name: a file name to open
    :param mode: opening mode
    :param compression: compression type

    :type name: str
    :type mode: str
    :type compression: str

    Compression type can be ``None``, ``'gzip'``, ``'bzip2'``, ``'lzma'``,
    as well as ``'use_ext'``. If ``'use_ext'`` is specified, compression
    algorithm is defined by the extension of the given file name.
    """
    fp: Incomplete
    ctype: Incomplete
    fp_extra: Incomplete
    def __init__(self, name, mode: str = 'r', compression: Incomplete | None = None) -> None:
        """
        Constructor.
        """
    def open(self, name, mode: str = 'r', compression: Incomplete | None = None) -> None:
        """
        Open a file pointer. Note that a file is *always* opened in text
        mode. The method inherits its input parameters from the constructor
        of :class:`FileObject`.
        """
    def close(self) -> None:
        """
        Close a file pointer.
        """
    def get_compression_type(self, file_name) -> None:
        """
        Determine compression type for a given file using its extension.

        :param file_name: a given file name
        :type file_name: str
        """
    def __enter__(self) -> None:
        """
        'with' constructor.
        """
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: types.TracebackType | None) -> None:
        """
        'with' destructor.
        """
