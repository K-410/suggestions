# This optimization converts PythonTokenTypes into a named tuple and binds
# its named members directly to the PythonToken instances.

# This is done for several reasons:
# 1. Pickling and unpickling is faster because no time is spent on calling
#    user-defined constructors as is the case with python's Enum class.
# 2. Binding directly to PythonToken means we no longer use user-defined
#    hashing, resulting in much faster dict lookup and insertions.

# This unfortunately touches many modules due to the extensive usage.


from parso.python import token
from textension.utils import _named_index, inline_class
from sys import intern


class TokenType(tuple):
    __slotnames__ = []
    __slots__ = ()

    name            = _named_index(0)
    contains_syntax = _named_index(1)

    def __repr__(self):
        return f"TokenType({self.name})"


# Maps old PythonTokenTypes to the new underlying value.
conversion_map = {
    t: TokenType((intern(t.value.name), t.value.contains_syntax))
    for t in token.PythonTokenTypes
}


@inline_class(conversion_map.values())
class PythonTokenTypes(tuple):
    __slots__ = ()

    STRING         = _named_index(0)
    NUMBER         = _named_index(1)
    NAME           = _named_index(2)
    ERRORTOKEN     = _named_index(3)
    NEWLINE        = _named_index(4)
    INDENT         = _named_index(5)
    DEDENT         = _named_index(6)
    ERROR_DEDENT   = _named_index(7)
    FSTRING_STRING = _named_index(8)
    FSTRING_START  = _named_index(9)
    FSTRING_END    = _named_index(10)
    OP             = _named_index(11)
    ENDMARKER      = _named_index(12)


token.PythonTokenTypes = PythonTokenTypes


from jedi.api import completion
completion.PythonTokenTypes = PythonTokenTypes

from parso import grammar
grammar.PythonTokenTypes = PythonTokenTypes
grammar.PythonGrammar._token_namespace = PythonTokenTypes

from parso.pgen2 import grammar_parser
grammar_parser.PythonTokenTypes = PythonTokenTypes

from parso.python import diff
diff.PythonTokenTypes = PythonTokenTypes

diff.NEWLINE      = PythonTokenTypes.NEWLINE
diff.DEDENT       = PythonTokenTypes.DEDENT
diff.NAME         = PythonTokenTypes.NAME
diff.ERROR_DEDENT = PythonTokenTypes.ERROR_DEDENT
diff.ENDMARKER    = PythonTokenTypes.ENDMARKER

from parso.python import parser
parser.PythonTokenTypes = PythonTokenTypes
parser.NAME = PythonTokenTypes.NAME
parser.INDENT = PythonTokenTypes.INDENT
parser.DEDENT = PythonTokenTypes.DEDENT

parser.Parser._leaf_map = {
    conversion_map[k]: v for k, v in parser.Parser._leaf_map.items()
}

from parso.python import tokenize

tokenize.PythonTokenTypes = PythonTokenTypes

tokenize.STRING         = PythonTokenTypes.STRING
tokenize.NAME           = PythonTokenTypes.NAME
tokenize.NUMBER         = PythonTokenTypes.NUMBER
tokenize.OP             = PythonTokenTypes.OP
tokenize.NEWLINE        = PythonTokenTypes.NEWLINE
tokenize.INDENT         = PythonTokenTypes.INDENT
tokenize.DEDENT         = PythonTokenTypes.DEDENT
tokenize.ENDMARKER      = PythonTokenTypes.ENDMARKER
tokenize.ERRORTOKEN     = PythonTokenTypes.ERRORTOKEN
tokenize.ERROR_DEDENT   = PythonTokenTypes.ERROR_DEDENT
tokenize.FSTRING_START  = PythonTokenTypes.FSTRING_START
tokenize.FSTRING_STRING = PythonTokenTypes.FSTRING_STRING
tokenize.FSTRING_END    = PythonTokenTypes.FSTRING_END

tokenize.Token.__annotations__["type"] = PythonTokenTypes
