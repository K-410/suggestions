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

def apply():
    # This optimization adds things to the global namespace and must run
    # before several other ones.
    optimize_parser()
    optimize_diffparser()
    optimize_grammar_parse()


# This makes some heavy optimizations to parso's Parser and token types:
# Node constructors are removed, its members are assigned directly.
def optimize_parser():
    from parso.parser import InternalParseError, StackNode
    from parso.python.tree import Operator, Keyword, Name
    from parso.python.token import PythonTokenTypes
    from parso.python.tree import PythonNode, EndMarker
    from parso.python.parser import Parser
    from parso.parser import StackNode
    from parso import grammar
    from collections import defaultdict
    from itertools import repeat

    # Use default __hash__, because Enum doesn't. Go figure.
    PythonTokenTypes.__hash__ = object.__hash__

    # Grammar must be rebuilt because we used a better hash method.
    from .tools import _get_inference_state
    state = _get_inference_state()
    grammar._loaded_grammars.clear()
    state.grammar = state.latest_grammar = grammar.load_grammar()
    reserved_syntax = state.grammar._pgen_grammar.reserved_syntax_strings

    # User-defined initializers are too costly. Assign members manually.
    PythonNode.__init__ = object.__init__
    Operator.__init__ = object.__init__
    Keyword.__init__ = object.__init__
    Name.__init__ = object.__init__

    class StackNode:
        __slots__ = ("dfa", "nodes")

        # On-demand.
        nonterminal = _descriptor(attrgetter("dfa.from_rule"))

    def subclass(cls):
        mapping = {"__qualname__": cls.__qualname__, "__slots__": ()}
        new_cls = type(cls.__name__, (cls,), mapping)
        # Make the class pickle-able.
        globals()[cls.__name__] = new_cls
        return new_cls

    # Same as the parser's own '_leaf_map', but defaults to Operator.
    leaf_map = defaultdict(repeat(Operator).__next__)

    # Skip modifying the initializer for these classes, because jedi will
    # construct them elsewhere, and pulling that code is not worth it.
    skip_modify = {EndMarker, }

    # Remove initializer for these also.
    for from_cls, to_cls in Parser._leaf_map.items():
        # Skip this because it's used elsewhere.
        if to_cls in skip_modify:
            to_cls = subclass(to_cls)
        to_cls.__init__ = object.__init__
        leaf_map[from_cls] = to_cls

    NAME = PythonTokenTypes.NAME
    leaf_map[NAME] = Name
    node_map = Parser.node_map

    @override_method(Parser)
    def _add_token(self: Parser, token):
        type, value, start_pos, prefix = token

        cls = leaf_map[type]
        if value in reserved_syntax:
            if type is NAME:
                cls = Keyword
            type = reserved_syntax[value]

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
                    return
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



# Optimize DiffParser.update to skip identical lines.
def optimize_diffparser():
    from parso.python.diff import _get_debug_error_message, DiffParser
    from difflib import SequenceMatcher
    from itertools import count, compress
    from operator import ne

    @override_method(DiffParser)
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



# Optimizes Grammar.parse and InferenceState.parse by stripping away nonsense.
def optimize_grammar_parse():
    assert _use_new_interpreter
    from parso.grammar import Grammar, parser_cache, \
        try_to_save_module, load_module
    from parso.python.diff import DiffParser
    from parso.python.parser import Parser

    org_parse = Grammar.parse
    is_bpy_text = BpyTextBlockIO.__instancecheck__

    @override_method(Grammar)
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
            with open(path) as f:
                lines = ensure_blank_eol(f.readlines())

        try:
            cached = parser_cache[self._hashed][path]
        except KeyError:
            node = Parser(self._pgen_grammar, start_nonterminal="file_input").parse(
                tokens=self._tokenizer(lines))
        else:
            if cached.lines == lines:
                return cached.node  # type: ignore
            node = DiffParser(self._pgen_grammar, self._tokenizer, cached.node).update(
                cached.lines, lines)

        # Bpy text blocks should not be pickled.
        do_pickle = not is_bpy_text(file_io)
        try:
            try_to_save_module(self._hashed, file_io, node, lines,
                pickling=do_pickle, cache_path=Path(settings.cache_directory))
        except:
            import traceback
            traceback.print_exc()
        return node  # type: ignore

    # @override_method(api.InferenceState)
    def parse(self, *_, **kw):
        assert "file_io" in kw, kw
        return self.grammar.parse(kw["file_io"])
    api.InferenceState.parse = parse

    set_optimized()
