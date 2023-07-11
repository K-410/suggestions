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
                nodes += [new_node]
            else:
                self.error_recovery(token)
                return None

        tos.dfa = plan.next_dfa

        for dfa, node in zip(plan.dfa_pushes, stack_nodes):
            node.dfa = dfa
            node.nodes = []
            stack += [node]

        leaf = new(cls)
        leaf.value = value
        leaf.line, leaf.column = start_pos
        leaf.prefix = prefix

        stack[-1].nodes += [leaf]
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
                raise Exception("InternalParseError")
                # raise InternalParseError(
                #     "incomplete input", t.type, t.string, t.start_pos)
            if len(self.stack) > 1:
                self._pop()
            else:
                return self.convert_node(tos.dfa.from_rule, tos.nodes)

    BaseParser.parse = parse

    StackNode.nonterminal = _forwarder("dfa.from_rule")


# Optimize DiffParser.update to skip identical lines.
def optimize_diffparser():
    from parso.python.diff import _get_debug_error_message, DiffParser
    from itertools import count, compress
    from textension.fast_seqmatch import FastSequenceMatcher
    from difflib import SequenceMatcher
    from operator import ne

    def update(self: DiffParser, old, new):
        self._module._used_names = None
        self._parser_lines_new = new
        self._reset()

        # The SequenceMatcher is slow for large files. This skips feeding
        # identical lines to it and instead apply offsets in the opcode.
        nlines = len(new)
        s = next(compress(count(), map(ne, new, old)), nlines - 1)
        sm = FastSequenceMatcher((old[s:], self._parser_lines_new[s:]))

        # Bump the opcode indices by equal lines.
        opcodes = [('equal', 0, s, 0, s)] + [
            (op, *map(s.__add__, indices)) for op, *indices in sm.get_opcodes()]

        for operation, i1, i2, j1, j2 in opcodes:
            if j2 == nlines and new[-1] == '':
                j2 -= 1
            if operation == 'equal':
                self._copy_from_old_parser(j1 - i1, i1 + 1, i2, j2)
            elif operation == 'replace':
                self._parse(until_line=j2)
            elif operation == 'insert':
                self._parse(until_line=j2)
            else:
                assert operation == 'delete'

        self._nodes_tree.close()
        last_pos = self._module.end_pos[0]
        assert last_pos == nlines, f"{last_pos} != {nlines}" + \
            _get_debug_error_message(self._module, old, new)
        return self._module
    
    DiffParser.update = update


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
            return Parser(self._pgen_grammar, start_nonterminal="file_input").parse(tokens=self._tokenizer(lines))

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


def optimize_tokenize_lines():
    from parso.python.tokenize import BOM_UTF8_STRING, PythonTokenTypes, FStringNode, \
        split_lines, _close_fstring_if_necessary, _get_token_collection, _split_illegal_unicode_name, tokenize_lines as _tokenize_lines
    from textension.utils import inline, _TupleBase, _named_index
    from .interpreter import PythonTokenTypes
    from itertools import chain, count, repeat
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

    class PythonToken2(_TupleBase):
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
    def strlen(string: str) -> int: return str.__len__

    from keyword import kwlist, softkwlist

    known_base = set(kwlist + softkwlist) | __builtins__.keys()

    operators = {"**", "**=", ">>", ">>=", "<<", "<<=", "//", "//=", "->",
                 "+", "+=", "-", "-=", "*", "*=", "/", "/=", "%", "%=", "&",
                 "&=", "@", "@=", "|", "|=", "^", "^=", "!", "!=", "=", "==",
                 "<", "<=", ">", ">=", "~", ":", ".", ",", ":="}

    non_letters = {"(", ")", "[", "]", "{", "}", "\n", "\r", "#", "0", "1",
                   "2", "3", "4", "5", "6", "7", "8", "9", "\"", "\'", "*",
                   ">", "<", "=", "/", "-", "+", "%", "&", "~", ":", ".", ",",
                   "!", "|", "^", "\\", "@"}
    from sys import intern

    def tokenize_lines(lines, *, version_info, indents=None, start_pos=(1, 0), is_first_token=True):
        # Materialize if lines is a generator/iterable.
        lines = list(lines)
        if not lines:
            return ()

        known = known_base.copy()
        add_known = known.add

        result = []
        if indents is None:
            indents = [0]

        pseudo_token, single_quoted, triple_quoted, endpats, whitespace, fstring_pattern_map, always_break_tokens = _get_token_collection(version_info)

        paren_level = 0  # count parentheses

        contstr  = ""
        contline = ""
        prefix   = ""
        additional_prefix = ""

        contstr_start = None
        endprog  = None
        token    = None
        endmatch = None
        new_line = True
        lnum = start_pos[0] - 1

        fstring_stack = []

        pos = 0
        end = 0

        if is_first_token:
            line = lines[0]
            if line[:1] == "\ufeff":
                additional_prefix = "\ufeff"
                line = line[1:]
            if start_pos[1]:
                line = "^" * start_pos[1] + line
            lines[0] = line

        for lnum, pos, end, line in zip(count(start_pos[0]), repeat(pos), map(strlen, lines), lines):
            if contstr:
                if endmatch := get_match(endprog, line):
                    pos = get_end(endmatch, 0)
                    result += ((STRING, contstr + line[:pos], contstr_start, prefix),)
                    contstr  = ''
                    contline = ''
                else:
                    contstr  += line
                    contline += line
                    continue

            while pos < end:
                if fstring_stack:
                    tos = fstring_stack[-1]
                    if not tos.is_in_expr():
                        string, pos = _find_fstring_string(endpats, fstring_stack, line, lnum, pos) # type: ignore
                        if string:
                            result += ((FSTRING_STRING, string, tos.last_string_start_pos, ''),)
                            tos.previous_lines = ''
                            continue
                        if pos == end:
                            break

                    fstring_end_token, additional_prefix, quote_length = _close_fstring_if_necessary(
                        fstring_stack, line[pos:], lnum, pos, additional_prefix) # type: ignore
                    pos += quote_length
                    if fstring_end_token:
                        result += (fstring_end_token,)
                        continue

                    string_line = line
                    for fstring_stack_node in fstring_stack:
                        quote = fstring_stack_node.quote
                        if end_match := get_match(endpats[quote], line, pos):
                            end_match_string = get_group(end_match, 0)
                            if strlen(end_match_string) - strlen(quote) + pos < strlen(string_line):
                                string_line = line[:pos] + end_match_string[:-strlen(quote)]
                    pseudomatch = get_match(pseudo_token, string_line, pos)
                else:
                    pseudomatch = get_match(pseudo_token, line, pos)

                if pseudomatch:
                    last_pos = pos
                    start, pos = get_span(pseudomatch, 2)
                    token = line[start:pos]

                    prefix = additional_prefix
                    if start and line[start - 1] in {" ", "\f", "\t"}:
                        prefix += line[last_pos:start]

                    additional_prefix = ""

                    if token == "":
                        assert prefix
                        additional_prefix = prefix
                        break
                    initial = token[0]
                else:
                    match = get_match(whitespace, line, pos)
                    start = pos = get_end(match)
                    initial = line[start]

                spos = (lnum, start)

                any.ncalls(new_line)
                if new_line and initial not in {"\r", "\n", "#"} and (initial is not "\\" or not pseudomatch):
                    new_line = False
                    if paren_level == 0 and not fstring_stack:
                        if start > indents[-1]:
                            result  += ((INDENT, '', spos, ''),)
                            indents += (start,)

                        else:
                            while start < indents[-1]:
                                if start > indents[-2]:
                                    result += ((ERROR_DEDENT, '', (lnum, start), ''),)
                                    indents[-1] = start
                                    break
                                del indents[-1]
                                result += ((DEDENT, '', spos, ''),)

                if not pseudomatch:  # scan for tokens
                    if new_line and paren_level == 0 and not fstring_stack:
                        while pos < indents[-1]:
                            if pos > indents[-2]:
                                result += ((ERROR_DEDENT, '', (lnum, pos), ''),)
                                indents[-1] = pos
                                break
                            del indents[-1]
                            result += ((DEDENT, '', spos, ''),)

                    new_line = False
                    result += ((ERRORTOKEN, line[pos], (lnum, pos), additional_prefix + get_group(match, 0)),)
                    additional_prefix = ''
                    pos += 1

                # Check ascii before group.
                # elif initial in asciis or get_group(pseudomatch, 3):
                elif initial not in non_letters and (token in known or (isidentifier(token) and not add_known(token))):
                    if token in always_break_tokens:
                        if fstring_stack or paren_level:
                            del fstring_stack[:]
                            paren_level = 0
                            if m := re.match(r'[ \f\t]*$', line[:start]):
                                indent_start = get_end(m)
                                while indent_start < indents[-1]:
                                    if indent_start > indents[-2]:
                                        result += ((ERROR_DEDENT, '', (lnum, indent_start), ''),)
                                        indents[-1] = indent_start
                                        break
                                    del indents[-1]
                                    result += ((DEDENT, '', spos, ''),)
                    # if token in known_names or (isidentifier(token) and not add_known(token)):
                    result += ((NAME, token, spos, prefix),)
                    # else:
                    #     result += _split_illegal_unicode_name(token, spos, prefix) # type: ignore

                elif initial is "\n":
                    if fstring_stack and any(not f.allow_multiline() for f in fstring_stack):
                        fstring_stack.clear()

                    if not new_line and paren_level == 0 and not fstring_stack:
                        result += ((NEWLINE, token, spos, prefix),)
                    else:
                        additional_prefix = prefix + token
                    new_line = True

                elif token in operators:
                    if fstring_stack and token[0] is ":" and fstring_stack[-1].parentheses_count - fstring_stack[-1].format_spec_count == 1:
                        fstring_stack[-1].format_spec_count += 1
                        token = ':'
                        pos = start + 1

                    result += ((OP, token, spos, prefix),)

                elif token in {"(", ")", "[", "]", "{", "}"}:
                    if token in {"(", "[", "{"}:
                        if fstring_stack:
                            fstring_stack[-1].open_parentheses(token)
                        else:
                            paren_level += 1
                    else:
                        if fstring_stack:
                            fstring_stack[-1].close_parentheses(token)
                        else:
                            if paren_level:
                                paren_level -= 1
                    result += ((OP, token, spos, prefix),)

                elif initial is "#":
                    if fstring_stack and fstring_stack[-1].is_in_expr():
                        result += ((ERRORTOKEN, initial, spos, prefix),)
                        pos = start + 1
                    else:
                        additional_prefix = prefix + token

                elif initial in {"\"", "\'"}:
                    if token in triple_quoted:
                        endprog = endpats[token]
                        endmatch = get_match(endprog, line, pos)
                        if endmatch:                                # all on one line
                            pos = get_end(endmatch, 0)
                            token = line[start:pos]
                            result += ((STRING, token, spos, prefix),)
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
                        result += ((STRING, token, spos, prefix),)

                elif (initial in {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"} or \
                     (initial == '.' != token != '...')
                ):
                    result += ((NUMBER, token, spos, prefix),)
                elif token in fstring_pattern_map:  # The start of an fstring.
                    fstring_stack += (FStringNode(fstring_pattern_map[token]),)
                    result += ((FSTRING_START, token, spos, prefix),)
                elif initial == '\\' and line[start:] in ('\\\n', '\\\r\n', '\\\r'):  # continued stmt
                    additional_prefix += prefix + line[start:]
                    break
                else:
                    assert False

        if contstr:
            result += ((ERRORTOKEN, contstr, contstr_start, prefix),)
            if contstr.endswith('\n') or contstr.endswith('\r'):
                new_line = True

        if fstring_stack:
            tos = fstring_stack[-1]
            if tos.previous_lines:
                result += ((FSTRING_STRING, tos.previous_lines, tos.last_string_start_pos, ''),)

        end_pos = lnum, end

        # XXX: Apparently this needs to run at the end to synchronize indents.
        def epilog():
            for indent in indents[1:]:
                indents.pop()
                yield PythonToken2((DEDENT, '', end_pos, ''))
            yield PythonToken2((ENDMARKER, '', end_pos, additional_prefix))

        # result += [(DEDENT, '', end_pos, '')] * (len(indents) - 1)
        # result += ((ENDMARKER, '', end_pos, additional_prefix),)
        return chain(map(PythonToken2, result), epilog())

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
# class TokenType(tuple):
#     __slotnames__ = []
#     __slots__ = ()

#     name            = _named_index(0)
#     contains_syntax = _named_index(1)

#     def __repr__(self):
#         return f"TokenType({self.name})"


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
