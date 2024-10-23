class MainThread:
    """
    A dummy class for checking whether the current thread is the main one.
    This is currently necessary for proper signal handling when making
    oracle calls and creating cardinality encodings.
    """
    @staticmethod
    def check() -> None:
        """
        The actual checker.
        """
