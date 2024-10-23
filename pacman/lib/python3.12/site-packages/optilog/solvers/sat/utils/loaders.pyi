from typing import NamedTuple

class SatSolverOutput(NamedTuple):
    model: list[int] | None
    sat: str | None
    def verify_on(self, formula) -> None: ...

class MaxSatSolverOutput(NamedTuple):
    model: list[int] | None
    sat: str | None
    cost: int | None
    def verify_on(self, formula) -> None: ...

def load_sat_output(path, model_format) -> None: ...
def load_maxsat_output(path, model_format) -> None: ...
