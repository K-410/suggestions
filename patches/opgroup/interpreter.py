# This adds optimizations to the various parsers.
#
# Add token optimization:
# - Eliminate token initializers and use direct assignments.
# - Use heuristics, reordering and class maps.
# - Use object hashing for PythonTokenType enums.
#
# DiffParser optimization:
# - Skip feeding identical lines to the SequenceMatcher
#
# Grammar.parse optimization:
# - Just removes junk code.
from textension.utils import _forwarder, _patch_function


_use_new_interpreter = []


def apply():
    optimize_tokens()
    optimize_parser()
    optimize_diffparser()
    optimize_grammar_parse()
    optimize_tokenize_lines()
    _use_new_interpreter[:] = [1]


# This makes some heavy optimizations to parso's Parser and token types:
# Node constructors are removed, its members are assigned directly.
def optimize_parser():
    from parso.python import parser, token, tree
    from parso import grammar

    from parso.parser import InternalParseError, StackNode
    from parso.python.tree import Keyword, PythonNode

    from collections import defaultdict
    from itertools import repeat

    from ..tools import state

    # Reload grammar since they depend on token hash.
    grammar._loaded_grammars.clear()
    state.grammar = state.latest_grammar = grammar.load_grammar()

    NAME = token.PythonTokenTypes.NAME
    reserved = state.grammar._pgen_grammar.reserved_syntax_strings

    # Same as the parser's own ``_leaf_map``, but defaults to Operator.
    leaf_map = defaultdict(repeat(tree.Operator).__next__, parser.Parser._leaf_map)
    leaf_map[NAME] = tree.Name

    node_map = parser.Parser.node_map
    new = object.__new__

    # Delete recovery tokenize. We inline it in ``_add_token`` instead.
    from parso.python.parser import DEDENT, INDENT, Parser
    del Parser.parse

    stack_nodes = map(new, repeat(StackNode))

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

        cls = leaf_map[type]
        if value in reserved:
            if type is NAME:
                cls = Keyword
            type = reserved[value]

        stack = self.stack
        tos = stack[-1]
        nodes = tos.nodes

        while True:
            dfa = tos.dfa
            if type in dfa.transitions:
                plan = dfa.transitions[type]
                break
            elif dfa.is_final:
                nonterminal = dfa.from_rule
                try:
                    new_node, = nodes
                except:
                    if nonterminal not in node_map:
                        if nonterminal == 'suite':
                            nodes = [nodes[0]] + nodes[2:-1]

                        # Bypass __init__ and assign directly.
                        new_node = new(PythonNode)
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
            else:
                self.error_recovery(token)
                return None

        tos.dfa = plan.next_dfa

        for dfa, node in zip(plan.dfa_pushes, stack_nodes):
            node.dfa = dfa
            node.nodes = []
            stack += node,

        leaf = new(cls)
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
    from textension.fast_seqmatch import FastSequenceMatcher
    from parso.python.diff import _get_debug_error_message, DiffParser
    from itertools import count, compress
    from builtins import map, next, reversed
    from operator import ne

    lstlen = list.__len__

    def update(self: DiffParser, a, b):
        self._module._used_names = None
        self._parser_lines_new = b
        self._reset()

        opcodes = []

        alen = lstlen(a)
        blen = lstlen(b)

        # Valid head (a and b are equal up to this).
        head = next(compress(count(), map(ne, a, b)), blen - 1)

        # Valid tail (a and b are equal from this to end).
        tail = next(compress(count(), map(ne, reversed(a), reversed(b))), 0)

        old_end = alen - tail
        new_end = blen - tail

        head = min(head, old_end, new_end)

        if head != 0:
            opcodes += ("equal", 0, head, 0, head),

        # Feed only changed lines, then add the offsets to the opcode indices.
        add_offset = head.__add__
        sm = FastSequenceMatcher((a[head:old_end], b[head:new_end]))

        for opcode, *indices in sm.get_opcodes():
            opcodes += (opcode, *map(add_offset, indices)),

        if tail != 0:
            j1, j2 = opcodes[-1][2::2]
            opcodes += ("equal", j1, alen, j2, blen),

        for op, i1, i2, j1, j2 in opcodes:
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

        # Jedi will sometimes parse small snippets as docstring modules.
        if not path and not file_io:
            lines = ensure_blank_eol(code.splitlines(True))
            # ``start_symbol`` can sometimes be ``eval_input``. The difference
            # is the type of root node Parser.parse returns.
            return Parser(self._pgen_grammar, start_nonterminal=start_symbol or self._start_nonterminal).parse(tokens=self._tokenizer(lines))

        if not path:
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


def optimize_tokenize_lines() -> None:
    from parso.python.tokenize import FStringNode, _close_fstring_if_necessary, \
        split_lines, _get_token_collection, tokenize_lines as _tokenize_lines, _find_fstring_string
    from textension.utils import inline, Aggregation, _named_index
    from .interpreter import PythonTokenTypes
    from itertools import chain, count, repeat
    from operator import attrgetter
    import re


    STRING         = PythonTokenTypes.STRING
    NAME           = PythonTokenTypes.NAME
    NUMBER         = PythonTokenTypes.NUMBER
    OP             = PythonTokenTypes.OP
    NEWLINE        = PythonTokenTypes.NEWLINE
    INDENT         = PythonTokenTypes.INDENT
    DEDENT         = PythonTokenTypes.DEDENT
    ENDMARKER      = PythonTokenTypes.ENDMARKER
    ERRORTOKEN     = PythonTokenTypes.ERRORTOKEN
    ERROR_DEDENT   = PythonTokenTypes.ERROR_DEDENT
    FSTRING_START  = PythonTokenTypes.FSTRING_START
    FSTRING_STRING = PythonTokenTypes.FSTRING_STRING
    FSTRING_END    = PythonTokenTypes.FSTRING_END

    class PythonToken2(Aggregation):
        type: PythonTokenTypes      = _named_index(0)
        string: str                 = _named_index(1)
        start_pos: tuple[int, int]  = _named_index(2)
        prefix: str                 = _named_index(3)

        @property
        def end_pos(self) -> tuple[int, int]:
            lines = split_lines(self.string)
            if len(lines) > 1:
                return self.start_pos[0] + len(lines) - 1, 0
            else:
                return self.start_pos[0], self.start_pos[1] + len(self.string)

    @inline
    def get_match(*args) -> re.Match: return re.Pattern.match
    @inline
    def get_end(*args) -> int: return re.Match.end
    @inline
    def get_group(*args) -> str: return re.Match.group
    @inline
    def get_span(*args) -> tuple[int, int]: return re.Match.span
    @inline
    def isidentifier(string: str) -> bool: return str.isidentifier
    @inline
    def startswith(string: str) -> bool: return str.startswith
    @inline
    def strlen(string: str) -> int: return str.__len__
    @inline
    def is_multiline(quote: str) -> bool: return {'"""', "'''"}.__contains__
    @inline
    def lstrip(string: str) -> str: return str.lstrip

    from keyword import kwlist, softkwlist

    known_base = set(kwlist + softkwlist) | __builtins__.keys()

    operators = {"**", "**=", ">>", ">>=", "<<", "<<=", "//", "//=", "->",
                 "+", "+=", "-", "-=", "*", "*=", "/", "/=", "%", "%=", "&",
                 "&=", "@", "@=", "|", "|=", "^", "^=", "!", "!=", "=", "==",
                 "<", "<=", ">", ">=", "~", ":", ".", ",", ":=", "(", ")", "[", "]", "{", "}"}

    non_letters = {"(", ")", "[", "]", "{", "}", "\n", "\r", "#", "0", "1",
                   "2", "3", "4", "5", "6", "7", "8", "9", "\"", "\'", "*",
                   ">", "<", "=", "/", "-", "+", "%", "&", "~", ":", ".", ",",
                   "!", "|", "^", "\\", "@", " "}

    # Triple quoted fstrings.
    triples = {
        "f\'\'\'", "F\'\'\'", "f\"\"\"", "F\"\"\"", "fr\'\'\'",
        "fR\'\'\'", "Fr\'\'\'", "FR\'\'\'", "fr\"\"\"", "fR\"\"\"",
        "Fr\"\"\"", "FR\"\"\"", "rf\'\'\'", "rF\'\'\'", "Rf\'\'\'",
        "RF\'\'\'", "rf\"\"\"", "rF\"\"\"", "Rf\"\"\"", "RF\"\"\"",
    }

    rep_zeros = repeat(0)
    get_quotes = attrgetter("quote")

    def tokenize_lines(lines, *, version_info, indents=None, start_pos=(1, 0), is_first_token=True):

        if indents is None:
            indents = [0]

        def dedent_if_necessary(start):
            nonlocal result, indents
            yield from map(PythonToken2, result)
            result = []

            while start < indents[-1]:
                if start > indents[-2]:
                    yield PythonToken2((ERROR_DEDENT, "", (lnum, start), ""))
                    indents[-1] = start
                    break
                del indents[-1]
                yield PythonToken2((DEDENT, "", spos, ""))

        known = known_base.copy()
        add_known = known.add

        pseudo_token, single_quoted, triple_quoted, endpats, whitespace, fstring_pattern_map, always_break_tokens = _get_token_collection(version_info)

        all_triples = triples | triple_quoted

        paren_level = 0  # count parentheses

        contstr  = ""
        contline = ""
        prefix   = ""
        line     = ""
        additional_prefix = ""

        contstr_start = 0
        endprog = None
        token = ""
        endmatch = None
        new_line = True
        lnum = start_pos[0] - 1

        result = []
        fstack = []

        positions = rep_zeros
        if is_first_token:
            if "\ufeff" in lines[0][:1]:
                lines[0] = lines[0][1:]
                additional_prefix = "\ufeff"

            if start_pos[1] != 0:
                positions = chain((start_pos[1],), rep_zeros)
                lines[0] = "^" * start_pos[1] + lines[0]

        lnum = 0
        pos  = 0
        end  = 0

        for lnum, pos, end, line in zip(count(start_pos[0]), positions, map(strlen, lines), lines):
            if contstr:
                if endmatch := get_match(endprog, line):
                    pos = get_end(endmatch, 0)
                    result += (STRING, contstr + line[:pos], contstr_start, prefix),
                    contstr  = ""
                    contline = ""
                else:
                    contstr  += line
                    contline += line
                    continue

            while pos < end:
                if fstack:
                    tos = fstack[-1]
                    if not tos.is_in_expr():
                        string, pos = _find_fstring_string(endpats, fstack, line, lnum, pos)
                        if string:
                            result += (FSTRING_STRING, string, tos.last_string_start_pos, ""),
                            tos.previous_lines = ""
                            continue
                        if pos == end:
                            break

                    fstring_end_token, additional_prefix, quote_length = _close_fstring_if_necessary(
                        fstack, line[pos:], lnum, pos, additional_prefix)
                    pos += quote_length
                    if fstring_end_token:
                        result += fstring_end_token,
                        continue

                    string_line = line
                    for fstring_stack_node in fstack:
                        quote = fstring_stack_node.quote
                        if end_match := get_match(endpats[quote], line, pos):
                            end_match_string = get_group(end_match, 0)
                            if strlen(end_match_string) - strlen(quote) + pos < strlen(string_line):
                                string_line = line[:pos] + end_match_string[:-strlen(quote)]
                    pseudomatch = get_match(pseudo_token, string_line, pos)
                else:
                    c = line[pos]
                    if c not in non_letters:  # Short circuit comparison.
                        pass
                    else:
                        if c is "\n":
                            if fstack and False in map(is_multiline, map(get_quotes, fstack)):
                                fstack = []
                            if not new_line and paren_level is 0 and not fstack:
                                result += (NEWLINE, "\n", (lnum, pos), prefix),
                            else:
                                additional_prefix += "\n"
                            new_line = True
                            break

                        elif c in {".", "=", ","}:
                            try:
                                d = c is not line[pos + 1]
                            except:
                                d = True

                            if d:
                                result += (OP, c, (lnum, pos), additional_prefix),
                                additional_prefix = ""
                                pos += 1
                                continue

                    pseudomatch = get_match(pseudo_token, line, pos)

                if pseudomatch:
                    last_pos = pos
                    start, pos = get_span(pseudomatch, 2)
                    token = line[start:pos]

                    prefix = additional_prefix
                    if start and line[start - 1] in {" ", "\f", "\t"}:
                        prefix += line[last_pos:start]

                    additional_prefix = ""

                    if token is "":
                        additional_prefix = prefix
                        break

                    initial = token[0]
                else:
                    match = get_match(whitespace, line, pos)
                    start = pos = get_end(match)
                    initial = line[start]

                spos = (lnum, start)

                if new_line and initial not in {"\r", "\n", "#", "\\"}:
                    new_line = False
                    if start != indents[-1] and paren_level is 0 and not fstack:
                        if start > indents[-1]:
                            # Only yield DEDENT straight away. The diff
                            # parser doesn't care about indent sync.
                            result += (INDENT, "", spos, ""),
                            indents += (start,)
                        elif start < indents[-1]:
                            yield from dedent_if_necessary(start)

                if not pseudomatch:
                    if start != indents[-1] and paren_level is 0 and not fstack and start < indents[-1]:
                        yield from dedent_if_necessary(start)
                    new_line = False
                    result += (ERRORTOKEN, line[pos], spos, additional_prefix + get_group(match, 0)),
                    additional_prefix = ""
                    pos += 1
                    continue

                elif initial not in non_letters and (token in known or (isidentifier(token) and not add_known(token))):
                    if token in always_break_tokens and (fstack or paren_level):
                        fstack = []
                        paren_level = 0
                        if m := re.match(r'[ \f\t]*$', line[:start]):
                            yield from dedent_if_necessary(get_end(m))
                    result += (NAME, token, spos, prefix),

                elif initial in {"\n", "\r"}:
                    if fstack and False in map(is_multiline, map(get_quotes, fstack)):
                        fstack = []
                    if not new_line and paren_level is 0 and not fstack:
                        result += (NEWLINE, token, spos, prefix),
                    else:
                        additional_prefix = prefix + token
                    new_line = True

                elif token in operators:
                    if fstack and token[0] is ":" and fstack[-1].parentheses_count - fstack[-1].format_spec_count is 1:
                        fstack[-1].format_spec_count += 1
                        token = ':'
                        pos = start + 1
                    elif token in {"(", ")", "[", "]", "{", "}"}:
                        if token in {"(", "[", "{"}:
                            if fstack:
                                fstack[-1].open_parentheses(token)
                            else:
                                paren_level += 1
                        else:
                            if fstack:
                                fstack[-1].close_parentheses(token)
                            else:
                                if paren_level:
                                    paren_level -= 1
                    result += (OP, token, spos, prefix),

                elif initial is "#":
                    if fstack and fstack[-1].is_in_expr():
                        result += (ERRORTOKEN, initial, spos, prefix),
                        pos = start + 1
                    else:
                        additional_prefix = prefix + token

                elif token in fstring_pattern_map:  # The start of an fstring.
                    fstack += (FStringNode(fstring_pattern_map[token]),)
                    result += (FSTRING_START, token, spos, prefix),

                elif initial in {"\"", "\'"}:
                    if token in triple_quoted:
                        endprog = endpats[token]
                        endmatch = get_match(endprog, line, pos)
                        if endmatch:                                # all on one line
                            pos = get_end(endmatch, 0)
                            token = line[start:pos]
                            result += (STRING, token, spos, prefix),
                        else:
                            contstr_start = spos                    # multiple lines
                            contstr = line[start:]
                            contline = line
                            break
                    elif token[-1] is "\n":
                        contstr_start = spos
                        endprog = endpats.get(initial) or endpats.get(token[1]) or endpats.get(token[2])
                        contstr = line[start:]
                        contline = line
                        break
                    else:                                       # ordinary string
                        result += (STRING, token, spos, prefix),

                elif (initial in {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"} or \
                   (initial is "." != token != "...")):
                    result += (NUMBER, token, spos, prefix),

                elif initial is '\\' and line[start:] in {"\\\n", "\\\r\n", "\\\r"}:  # continued stmt
                    additional_prefix += prefix + line[start:]
                    break

                elif token in all_triples:
                    if token in triple_quoted:
                        endprog = endpats[token]
                        if endmatch := get_match(endprog, line, pos):
                            pos = get_end(endmatch, 0)
                            token = line[start:pos]
                            result += (STRING, token, spos, prefix),
                        else:
                            contstr_start = spos                    # multiple lines
                            contstr = line[start:]
                            contline = line
                            break
                    else:
                        fstack += (FStringNode(fstring_pattern_map[token]),)
                        result += (FSTRING_START, token, spos, prefix),


                else:
                    if token not in  {"...", ";"}:
                        # TODO: Handle raw strings.
                        print("unhandled OP token:", repr(token), spos, prefix)
                    result += (OP, token, spos, prefix),

            if result:
                yield from map(PythonToken2, result)
                result = []

        if contstr:
            result += (ERRORTOKEN, contstr, contstr_start, prefix),

        if fstack:
            tos = fstack[-1]
            if tos.previous_lines:
                result += (FSTRING_STRING, tos.previous_lines, tos.last_string_start_pos, ""),

        yield from map(PythonToken2, result)

        end_pos = lnum, end

        for _ in indents[1:]:
            del indents[-1]
            yield PythonToken2((DEDENT, "", end_pos, ""))
        yield PythonToken2((ENDMARKER, "", end_pos, additional_prefix))

        return None

    _patch_function(_tokenize_lines, tokenize_lines)


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
