def get_color_tuples_to_consider() -> None: ...
def print_clauses_color(sets_of_literals: list[list[int]], clauses: list[list[int]], colors: str | list[str] = None, filter_clauses: bool = True):
    """
    Print clauses with literals colored according to the given colors.

    :param sets_of_literals: List of sets of literals, each set of literals is a list of literals
    :param clauses: List of clauses, each clause is a list of literals
    :param colors: If a string, the color to use for all the sets of literals. If a list of strings, the color to use for each set of literals. If None, the default colors will be used.
    :param filter_clauses: If True, only print clauses that contain at least one of the given literals

    """
