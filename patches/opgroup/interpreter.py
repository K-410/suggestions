"""This adds optimizations to the various parsers.

Add token optimization:
- Eliminate token initializers and use direct assignments.
- Use heuristics, reordering and class maps.
- Use object hashing for PythonTokenType enums.

DiffParser optimization:
- Skip feeding identical lines to the SequenceMatcher

Grammar.parse optimization:
- Just removes junk code.
"""
from textension.utils import _forwarder


_use_new_interpreter = []


def apply() -> None:
    optimize_tokens()
    optimize_parser()
    optimize_diffparser()
    optimize_grammar_parse()
    from . import tokenizer
    tokenizer.apply()
    _use_new_interpreter[:] = [1]


# This makes some heavy optimizations to parso's Parser and token types:
# Node constructors are removed, its members are assigned directly.
def optimize_parser():
    from parso.python.tree import Keyword, PythonNode
    from parso.python import parser, token, tree
    from parso.parser import StackNode
    from parso import grammar

    from collections import defaultdict
    from itertools import repeat
    from functools import partial
    from ..tools import state

    # Reload grammar since they depend on token hash.
    grammar._loaded_grammars.clear()
    state.grammar = state.latest_grammar = grammar.load_grammar()

    NAME = token.PythonTokenTypes.NAME
    reserved = state.grammar._pgen_grammar.reserved_syntax_strings

    new = object.__new__
    new_keyword  = partial(new, Keyword)
    new_pynode   = partial(new, PythonNode)
    new_operator = partial(new, tree.Operator)

    node_map = parser.Parser.node_map

    # Same as the parser's own ``_leaf_map``, but defaults to Operator.
    default_leaf = repeat(new_operator).__next__

    leaf_map = defaultdict(default_leaf)
    leaf_map.update(parser.Parser._leaf_map)
    leaf_map[NAME] = tree.Name

    for key, cls in tuple(leaf_map.items()):
        new_cls = partial(new, cls)
        leaf_map[key] = new_cls
    leaf_map[Keyword] = new_keyword

    # Remove recovery tokenize and instead inline it below.
    from parso.python.parser import DEDENT, INDENT, Parser
    del Parser.parse

    # Just an infinite stream of new StackNode objects.
    stack_nodes = map(new, repeat(StackNode))

    lstlen = list.__len__

    def _add_token(self: Parser, token):
        type, value, start_pos, prefix = token
        if type in {INDENT, DEDENT}:
            if type is DEDENT:
                self._indent_counter -= 1
                o = self._omit_dedent_list
                if o and o[-1] == self._indent_counter + 1:
                    del o[-1]
                    return
            else:
                self._indent_counter += 1

        new_cls = leaf_map[type]
        if value in reserved:
            if type is NAME:
                new_cls = leaf_map[Keyword]
            type = reserved[value]

        stack = self.stack
        tos = stack[-1]
        nodes = tos.nodes
        dfa = tos.dfa

        while type not in dfa.transitions:
            if dfa.is_final:
                if lstlen(nodes) is 1:
                    new_node = nodes[0]
                else:
                    nonterminal = dfa.from_rule
                    if nonterminal not in node_map:
                        if nonterminal == 'suite':
                            nodes = [nodes[0]] + nodes[2:-1]

                        # Bypass __init__ and assign directly.
                        new_node = new_pynode()
                        new_node.type = nonterminal
                        new_node.children = nodes
                        for child in nodes:
                            child.parent = new_node
                    else:
                        # Can't bypass initializer, some have custom ones.
                        new_node = node_map[nonterminal](nodes)
                del stack[-1]
                tos = stack[-1]
                nodes = tos.nodes
                nodes += new_node,
                dfa = tos.dfa
            else:
                self.error_recovery(token)
                return None

        plan = tos.dfa.transitions[type]
        tos.dfa = plan.next_dfa

        for dfa, node in zip(plan.dfa_pushes, stack_nodes):
            node.dfa = dfa
            node.nodes = []
            stack += node,

        leaf = new_cls()
        leaf.value = value
        leaf.line, leaf.column = start_pos
        leaf.prefix = prefix

        stack[-1].nodes += leaf,
        return None

    Parser._add_token = _add_token

    from parso.parser import BaseParser, Stack
    from textension.utils import consume

    def parse(self: BaseParser, tokens):
        node = StackNode(self._pgen_grammar.nonterminal_to_dfas[self._start_nonterminal][0])
        self.stack = Stack([node])
        consume(map(self._add_token, tokens))

        while True:
            tos = self.stack[-1]
            if not tos.dfa.is_final:
                raise Exception(f"InternalParseError, {self.stack[-1]}")
                # raise InternalParseError(
                #     "incomplete input", t.type, t.string, t.start_pos)
            if len(self.stack) > 1:
                self._pop()
            else:
                return self.convert_node(tos.dfa.from_rule, tos.nodes)

    BaseParser.parse = parse

    StackNode.nonterminal = _forwarder("dfa.from_rule")


# Optimize DiffParser to only parse the modified body, skipping the sequence
# matcher entirely.
def optimize_diffparser():
    from textension.fast_seqmatch import unified_diff
    from parso.python.diff import _get_debug_error_message, DiffParser

    def update(self: DiffParser, a, b):
        self._module._used_names = None
        self._parser_lines_new = b
        self._reset()
        blen = len(b)

        for op, i1, i2, j1, j2 in unified_diff(a, b):
            if j2 == blen and b[-1] == "":
                j2 -= 1
            if op == "equal":
                self._copy_from_old_parser(j1 - i1, i1 + 1, i2, j2)
            elif op in {"replace", "insert"}:
                self._parse(until_line=j2)

        self._nodes_tree.close()
        last_pos = self._module.end_pos[0]
        assert last_pos == blen, f"{last_pos} != {blen}" + _get_debug_error_message(self._module, a, b)
        return self._module
    
    DiffParser.update = update


# Optimizes Grammar.parse and InferenceState.parse by stripping away nonsense.
def optimize_grammar_parse():
    from parso.python.diff import DiffParser, Parser
    from jedi.inference import InferenceState
    from parso.grammar import Grammar, parser_cache, try_to_save_module, load_module
    from parso.utils import python_bytes_to_unicode
    from ..common import BpyTextBlockIO
    from ..tools import ensure_blank_eol
    from pathlib import Path

    def parse(self: Grammar,
              code,
              *,
              error_recovery=True,
              path=None,
              start_symbol=None,
              cache=False,
              diff_cache=False,
              cache_path=None,
              file_io: BpyTextBlockIO = None):

        if not path:
            # Jedi will sometimes parse small snippets as docstring modules.
            if not file_io:
                lines = ensure_blank_eol(code.splitlines(True))
                # ``start_symbol`` can sometimes be ``eval_input``. The difference
                # is the type of root node Parser.parse returns.
                return Parser(self._pgen_grammar, start_nonterminal=start_symbol or self._start_nonterminal).parse(tokens=self._tokenizer(lines))

            path = file_io.path

        if isinstance(cache_path, str):
            cache_path = Path(cache_path)

        # If jedi has already parsed and saved the stub, load from disk.
        if node := load_module(self._hashed, file_io, cache_path=cache_path):
            return node  # type: ignore

        # BpyTextBlockIO stores lines in self._content
        try:
            lines = file_io._content
        # Probably a generic FileIO
        except:
            with open(path, "rt") as f:
                lines = ensure_blank_eol(f.readlines())

        # Not ideal at all. Jedi loads some modules as bytes.
        # TODO: This should be intercepted in load_module.
        if isinstance(lines, bytes):
            lines = ensure_blank_eol(python_bytes_to_unicode(lines).splitlines(True))

        try:
            cached = parser_cache[self._hashed][path]
        except KeyError:
            node = Parser(self._pgen_grammar, start_nonterminal="file_input").parse(tokens=self._tokenizer(lines))
        else:
            if cached.lines == lines:
                return cached.node  # type: ignore
            node = DiffParser(self._pgen_grammar, self._tokenizer, cached.node).update(cached.lines, lines)

        # Bpy text blocks should not be pickled.
        do_pickle = not isinstance(file_io, BpyTextBlockIO)
        try:
            try_to_save_module(self._hashed, file_io, node, lines, pickling=do_pickle, cache_path=cache_path)
        except:
            import traceback
            traceback.print_exc()
        return node  # type: ignore
    
    Grammar.parse = parse


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


def _token_type_repr(self):
    return f"TokenType({self.name})"


def _create_token_type(name: str, contains_syntax: bool):
    return type("TokenType", (), {"name": name,
                                  "contains_syntax":contains_syntax,
                                  "__repr__": _token_type_repr,
                                  "__slots__": (),
                                  "__slotnames__": []})


# Maps old PythonTokenTypes to the new underlying value.
conversion_map = {
    t: _create_token_type(intern(t.value.name), t.value.contains_syntax)
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

def optimize_tokens():
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
