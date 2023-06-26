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


def apply():
    optimize_parser()
    optimize_diffparser()
    optimize_grammar_parse()


# This makes some heavy optimizations to parso's Parser and token types:
# Node constructors are removed, its members are assigned directly.
def optimize_parser():
    from parso.python import parser, token, tree
    from parso import grammar

    from parso.parser import InternalParseError, StackNode
    from parso.python.tree import Keyword, PythonNode

    from textension.utils import _forwarder
    from collections import defaultdict
    from itertools import repeat

    from ..tools import state

    # Use default __hash__ - Enum isn't suited for dictionaries.
    token.PythonTokenTypes.__hash__ = object.__hash__

    # Reload grammar since they depend on token hash.
    grammar._loaded_grammars.clear()
    state.grammar = state.latest_grammar = grammar.load_grammar()

    NAME = token.PythonTokenTypes.NAME
    reserved = state.grammar._pgen_grammar.reserved_syntax_strings

    # User-defined initializers are too costly. Assign members manually.
    tree.Keyword.__init__    = \
    tree.Name.__init__       = \
    tree.Operator.__init__   = \
    tree.PythonNode.__init__ = object.__init__

    class StackNode:
        __slots__ = ("dfa", "nodes")
        nonterminal = _forwarder("dfa.from_rule")  # Make on-demand.

    def subclass(cls):
        mapping = {"__qualname__": cls.__qualname__, "__slots__": ()}
        new_cls = type(cls.__name__, (cls,), mapping)

        # Make the class pickle-able.
        globals()[cls.__name__] = new_cls
        return new_cls

    # Same as the parser's own ``_leaf_map``, but defaults to Operator.
    leaf_map = defaultdict(repeat(tree.Operator).__next__)

    # Remove initializer for classes in ``_leaf_map``.
    for from_cls, to_cls in parser.Parser._leaf_map.items():
        # Used outside of ``_add_token``.
        if to_cls in {tree.EndMarker,}:
            to_cls = subclass(to_cls)

        to_cls.__init__ = object.__init__
        leaf_map[from_cls] = to_cls

    leaf_map[NAME] = tree.Name
    node_map = parser.Parser.node_map

    def _add_token(self: parser.Parser, token):
        type, value, start_pos, prefix = token

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
            try:
                plan = dfa.transitions[type]
                break

            except KeyError:
                if dfa.is_final:
                    nonterminal = dfa.from_rule
                    try:
                        new_node, = nodes
                    except:
                        if nonterminal in node_map:
                            new_node = node_map[nonterminal](nodes)
                        else:
                            if nonterminal == 'suite':
                                nodes = [nodes[0]] + nodes[2:-1]
                            new_node = PythonNode()
                            new_node.type = nonterminal
                            new_node.children = nodes
                            new_node.parent = None
                            for child in nodes:
                                child.parent = new_node

                    del stack[-1]
                    tos = stack[-1]
                    nodes = tos.nodes
                    nodes += [new_node]
                else:
                    self.error_recovery(token)
                    return None
            except IndexError:
                raise InternalParseError("too much input", type, value, start_pos)

        tos.dfa = plan.next_dfa

        for dfa in plan.dfa_pushes:
            node = StackNode()
            node.dfa = dfa
            node.nodes = []
            stack += [node]

        leaf = cls()
        leaf.value = value
        leaf.line, leaf.column = start_pos
        leaf.prefix = prefix

        stack[-1].nodes += [leaf]
        return None

    parser.Parser._add_token = _add_token


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
        sm = SequenceMatcher(None, old[s:], self._parser_lines_new[s:])

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


# Optimizes Grammar.parse and InferenceState.parse by stripping away nonsense.
def optimize_grammar_parse():
    from parso.python.diff import DiffParser, Parser
    from jedi.inference import InferenceState
    from parso.grammar import Grammar, parser_cache, try_to_save_module, load_module
    from parso.utils import python_bytes_to_unicode
    from ..common import BpyTextBlockIO, ensure_blank_eol
    from pathlib import Path
    from jedi import settings

    org_parse = Grammar.parse

    def parse(self: Grammar, file_io: BpyTextBlockIO, *_, **__):
        try:
            path = file_io.path
        except:
            # raise
            return org_parse(self, file_io, *_, **__)

        # XXX: Is path ever None?
        assert path is not None
        # XXX: This is for stubs.
        # If jedi has already parsed and saved the stub, load from disk.
        if path is not None:
            module_node = load_module(self._hashed, file_io, cache_path=Path(settings.cache_directory))
            if module_node is not None:
                return module_node  # type: ignore

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
            try_to_save_module(self._hashed, file_io, node, lines,
                pickling=do_pickle, cache_path=Path(settings.cache_directory))
        except:
            import traceback
            traceback.print_exc()
        return node  # type: ignore
    
    Grammar.parse = parse

    def parse(self: InferenceState, *_, **kw):
        assert "file_io" in kw, kw
        return self.grammar.parse(kw["file_io"])
    
    InferenceState.parse = parse
