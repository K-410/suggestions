# This adds optimizations for the Interpreter.

from jedi.inference.value.module import ModuleValue
from jedi.inference.context import GlobalNameFilter
from jedi.api.interpreter import MixedModuleContext, MixedParserTreeFilter
from jedi.inference import InferenceState
from jedi.api import Script, Completion

from ..common import BpyTextBlockIO, find_definition
from ..tools import ensure_blank_eol, state

from textension.utils import consume, _unbound_getter
from typing import TypeVar


Unused = TypeVar("Unused")
state_values = state.memoize_cache.values()


def apply():
    optimize_parser()
    optimize_diffparser()
    optimize_grammar_parse()


class BpyTextModuleValue(ModuleValue):
    inference_state = state  # Static
    parent_context  = None   # Static
    __init__ = object.__init__

    _is_package = False
    is_package  = bool

    # Overrides py__file__ to return the relative path instead of the
    # absolute one. On unix this doesn't matter. On Windows it erroneously
    # prefixes a drive to the beginning of the path:
    # '/MyText' -> 'C:/MyText'.
    py__file__ = _unbound_getter("_path")


class BpyTextModuleContext(MixedModuleContext):
    inference_state = state
    mixed_values = ()  # For mixed namespaces. Unused.

    is_stub = bool

    def __init__(self):
        self._value = BpyTextModuleValue()
        self.predefined_names = {}

    # Update the module value's runtime with the current interpreter.
    def update(self, interp):
        # TODO: These could all be forwarders to a ``self.interpreter``
        value = self._value
        value.tree_node = interp._module_node
        value.file_io = interp._file_io
        value.string_names = ("__main__",)
        value.code_lines = interp._code_lines
        value._path = value.file_io.path
        self.filters = None

    def get_filters(self, until_position=None, origin_scope=None):
        if not self.filters:
            # Skip the merged filter.
            # It's pointless since we're already returning A LIST OF FILTERS.
            tree_filter   = MixedParserTreeFilter(self, None, until_position, origin_scope)
            global_filter = GlobalNameFilter(self)
            self.filters  = [tree_filter, global_filter]
        return self.filters

    def py__getattribute__(self, name_or_str, name_context=None, position=None, analysis_errors=True):
        if namedef := find_definition(self, name_or_str, position):
            return namedef.infer()
        print("BpyTextModuleContext py__getattribute__ failed for", name_or_str)
        return super().py__getattribute__(name_or_str, name_context, position, analysis_errors)


class Interpreter(Script):
    _inference_state = state

    # Needed by self.get_signatures.
    _get_module_context = _unbound_getter("context")

    def __init__(self, code: str, _: Unused = None):
        # The memoize cache stores a dict of dicts keyed to functions.
        # The first level is never cleared. This lowers the cache overhead.
        consume(map(dict.clear, state_values))

        state.reset_recursion_limitations()
        state.inferred_element_counts = {}

        lines = ensure_blank_eol(code.splitlines(True))
        file_io = BpyTextBlockIO(lines)
        self._code_lines = lines
        self._file_io = file_io

        try:
            module_node = state.grammar.parse(file_io)
        # Support optimizations turned off
        except AttributeError:
            module_node = state.grammar.parse(code=code, file_io=file_io)

        self._module_node = module_node

        self.context = BpyTextModuleContext()
        self.context.update(self)

    def complete(self, line, column, *, fuzzy=False):
        return Completion(
            state,
            self.context,
            self._code_lines,
            (line, column),
            self.get_signatures,
            fuzzy=fuzzy
        ).complete()

    __repr__ = object.__repr__


# This adds optimizations to the various parsers Jedi uses.
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
# - Just removes irrelevant code.



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
    from parso.grammar import Grammar, parser_cache, try_to_save_module, load_module
    from parso.python.diff import DiffParser
    from parso.python.parser import Parser
    from parso.utils import python_bytes_to_unicode
    from pathlib import Path
    from jedi import settings

    org_parse = Grammar.parse
    is_bpy_text = BpyTextBlockIO.__instancecheck__
    is_bytes = bytes.__instancecheck__

    def parse(self: Grammar, file_io: BpyTextBlockIO, *_, **__):
        try:
            path = file_io.path
        except:
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
        if is_bytes(lines):
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
        do_pickle = not is_bpy_text(file_io)
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
