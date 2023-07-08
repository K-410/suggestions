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
from textension.utils import _forwarder


_use_new_interpreter = []


def apply():
    optimize_parser()
    optimize_diffparser()
    optimize_grammar_parse()
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
                #     "incomplete input", token.type, token.string, token.start_pos)
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
    from parso.python.diff import _get_debug_error_message, DiffParser
    from itertools import count, compress
    from builtins import map, next, reversed
    from operator import ne
    lstlen = list.__len__

    def update(self: DiffParser, a, b):
        self._module._used_names = None
        self._parser_lines_new = b
        self._reset()

        alen = lstlen(a)
        blen = lstlen(b)

        head = next(compress(count(), map(ne, a, b)), blen - 1)
        tail = next(compress(count(), map(ne, reversed(a), reversed(b))), 0)

        old_end = alen - tail
        new_end = blen - tail
        opcodes = []

        if head != 0:
            opcodes += [["equal", 0, head, 0, head]]

        opcodes += [["replace", head, old_end, head, new_end]]

        if alen != old_end != blen != new_end:
            opcodes += [["equal", old_end, alen, new_end, blen]]

        for op, i1, i2, j1, j2 in opcodes:
            if j2 == blen and b[-1] == "":
                j2 -= 1
            if op == "equal":
                self._copy_from_old_parser(j1 - i1, i1 + 1, i2, j2)
            elif op == "replace":
                self._parse(until_line=j2)

        self._nodes_tree.close()
        last_pos = self._module.end_pos[0]
        assert last_pos == blen, f"{last_pos} != {blen}" + \
            _get_debug_error_message(self._module, a, b)
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
