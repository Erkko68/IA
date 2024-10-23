from _typeshed import Incomplete as Incomplete
from sly import Lexer, Parser

class CalcLexer(Lexer):
    """
    Lexer for boolean expressions.
    """
    tokens: Incomplete
    ANDAL: str
    ORAL: str
    TRUE: str
    FALSE: str
    IMPLIES: str
    NAME: str
    NUMBER: str
    NOT: str
    AND: str
    XOR: str
    OR: str
    IF: str
    IFF: str
    EQUALS: str
    LTEQ: str
    LT: str
    GTEQ: str
    GT: str
    PLUS: str
    MINUS: str
    POWER: str
    TIMES: str
    DIVIDE: str
    LPAREN: str
    RPAREN: str
    ignore: str
    ignore_newline: str
    def ignore_newline(self, t) -> None: ...
    def error(self, t) -> None: ...

class ParsingException(Exception):
    lineno: Incomplete
    token: Incomplete
    def __init__(self, lineno, token) -> None: ...

class CalcParser(Parser):
    """
    CalcParser is a parser for boolean expressions.
    It is based on the sly library and uses the CalcLexer as a lexer.
    Takes into account the same precedence rules as Python.
    """
    tokens: Incomplete
    precedence: Incomplete
    expression: Incomplete
    def __init__(self) -> None: ...
    def statement(self, p) -> None: ...
    def error(self, token) -> None: ...

def parse_boolean_environment(text) -> None: ...
