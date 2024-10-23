from _typeshed import Incomplete as Incomplete

def sink(*args, **kwargs) -> None: ...

class LinearMaxSat:
    formula: Incomplete
    num_vars: Incomplete
    solver: Incomplete
    verbose: Incomplete
    weights: Incomplete
    opt_callback_func: Incomplete
    not_decide_vars: Incomplete
    decide_card_vars: Incomplete
    prev_model_: Incomplete
    def __init__(self, formula, solver: Incomplete | None = None, opt_callback_func: Incomplete | None = None, seed: int = 1234, not_decide_vars=[], decide_card_vars: bool = True, verbose: bool = True) -> None: ...
    block_vars: Incomplete
    best_model_: Incomplete
    best_cost: Incomplete
    def solve(self, assumptions=[], timeout: Incomplete | None = None, intermediate_timeout: Incomplete | None = None): ...
    def get_model(self) -> None: ...
