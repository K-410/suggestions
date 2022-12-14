# This module implements various optimizations for jedi and parso.


_version_compat_jedi = '0.18.1'
_version_compat_parso = '0.8.3'

from .tools import _deferred_patch_function, _get_super_meth

from jedi.api import Script
from jedi.api.project import Project
from jedi.api.environment import InterpreterEnvironment
from jedi.api.interpreter import MixedModuleContext
from jedi.api.completion import Completion
from jedi.inference import InferenceState
from jedi.inference.value.module import ModuleValue
from jedi.inference.imports import import_module_by_names

from jedi import settings
from pathlib import Path
from parso.file_io import KnownContentFileIO
from jedi.file_io import FileIOFolderMixin
import bpy


def setup2():
    import jedi
    import parso
    assert jedi.__version__ == _version_compat_jedi
    assert parso.__version__ == _version_compat_parso

    optimize_platform_system()
    optimize_name_get_definition()
    optimize_get_used_names()
    optimize_classfilter_filter()       # Call after optimize_parsertreefilter_filter
    optimize_get_definition_names()
    optimize_split_lines()
    optimize_fuzzy_match()
    optimize_filter_names()
    optimize_valuewrapperbase_getattr()
    optimize_lazyvaluewrapper_wrapped_value()

    optimize_getmodulename()
    optimize_get_parent_scope()
    optimize_get_cached_parent_scope()
    optimize_abstractusednamesfilter_values()
    optimize_defined_names()
    optimize_get_defined_names()
    optimize_stubfilter()
    # optimize_selfattributefilter_filter()
    # optimize_check_flows()
    # optimize_parsertreefilter_filter()  # Broken. Causes completions to disappear on second trigger.
    # optimize_valuewrapperbase_name()
    # optimize_parser()



class BpyTextBlockIO(KnownContentFileIO, FileIOFolderMixin):
    def __init__(self, text: bpy.types.Text, code):
        from pathlib import Path
        import os
        assert isinstance(text, bpy.types.Text), code
        self.path = Path(os.path.join(bpy.data.filepath, text.name))
        self._content = code
    def read(self):
        return self._content
    def get_last_modified(self):
        """Returns float - timestamp or None, if path doesn't exist."""
        # try:
        #     import os
        #     return os.path.getmtime(self.path)
        # except FileNotFoundError:
        #     return None

class BpyTextModuleValue(ModuleValue):
    def py__file__(self):
        return self._text_path


class Interpreter(Script):
    __slots__ = ()
    _inference_state = InferenceState(Project(Path("~")), InterpreterEnvironment(), None)

    def __new__(cls, *args, **kw):
        self = super().__new__(cls)
        cls._inference_state.allow_descriptor_getattr = True
        InferenceState.builtins_module, = import_module_by_names(
            self._inference_state, ("builtins",), sys_path=None, prefer_stubs=False)
        Interpreter.__new__ = lambda *args, **kw: self
        return self

    def __init__(self, code: str, text):
        assert isinstance(text, bpy.types.Text), type(text)

        # Workaround for jedi's recursion limit design.
        self._inference_state.inferred_element_counts.clear()
        self._code_lines = lines = code.splitlines(True) or [""]
        if code[-1] is "\n":
            lines.append("")
        self._file_io = BpyTextBlockIO(text, lines)
        self._module_node = self._inference_state.grammar.parse(
            code=code,
            diff_cache=True,
            cache_path=settings.cache_directory,
            file_io=self._file_io)
        self._code = code

    def _get_module_context(self):
        tree_module_value = BpyTextModuleValue(
            self._inference_state,
            self._module_node,
            file_io=self._file_io,
            string_names=('__main__',),
            code_lines=self._code_lines
        )
        tree_module_value._text_path = self._file_io.path
        return MixedModuleContext(tree_module_value, ())

    def complete(self, line, column, *, fuzzy=False):
        return Completion(
            self._inference_state,
            self._get_module_context(),
            self._code_lines,
            (line, column),
            self.get_signatures,
            fuzzy=fuzzy).complete()

    __repr__ = object.__repr__


def _clear_caches(_caches=[], _clear=dict.clear) -> None:
    """
    Internal.
    Evict all caches passed to, or obtained by, _register_cache().
    """
    for cache in _caches:
        _clear(cache)
    return None


def _register_cache(cache: dict = None) -> dict:
    if cache is None:
        cache = {}
    assert isinstance(cache, dict), f"Expected dict or None, got {type(cache)}"
    _clear_caches.__defaults__[0].append(cache)
    return cache


# This makes some heavy optimizations to parso's Parser and token types:
# Node constructors are removed, its members are assigned directly.
def optimize_parser():
    from parso.python.parser import Parser
    from parso.python.tree import Operator, Keyword, Name
    from parso.parser import InternalParseError, StackNode
    from parso.python.token import PythonTokenTypes
    from parso.parser import StackNode
    from parso.tree import BaseNode, Leaf
    from parso import grammar

    # Make PythonTokenTypes use object.__hash__.
    PythonTokenTypes.__hash__ = object.__hash__
    state = Interpreter._inference_state
    grammar._loaded_grammars.clear()
    state.grammar = state.latest_grammar = grammar.load_grammar(version='3.7')

    def remove_ctor(cls):
        mapping = {"__init__": object.__init__, "__slots__": ()}
        return type(cls.__name__, (cls,), mapping)

    # Use default object constructor.
    Operator = remove_ctor(Operator)
    Keyword  = remove_ctor(Keyword)
    Name     = remove_ctor(Name)

    class StackNode:
        __slots__ = ("dfa", "nodes")
        @property
        def nonterminal(self):
            return self.dfa.from_rule

    # Leaf map so the KeyError happens only once
    class LeafMap(dict):
        def __missing__(self, key):
            self[key] = Operator
            return Operator

    class PythonNode(BaseNode):
        __slots__ = ("type",)
        __init__ = object.__init__
        def __repr__(self):
            return "%s(%s, %r)" % (self.__class__.__name__, self.type, self.children)
        def get_name_of_position(self, position):
            for c in self.children:
                if isinstance(c, Leaf):
                    if c.type == 'name' and c.start_pos <= position <= c.end_pos:
                        return c
                else:
                    result = c.get_name_of_position(position)
                    if result is not None:
                        return result
            return None

    reserved_syntax_strings = state.grammar._pgen_grammar.reserved_syntax_strings
    leaf_map = LeafMap({k: remove_ctor(v) for k, v in Parser._leaf_map.items()})
    NAME = PythonTokenTypes.NAME
    node_map = Parser.node_map
    append = list.append

    def _add_token(self: Parser, token):
        stack = self.stack
        type_, value, start_pos, prefix = token
        transition = type_

        cls = Name if type_ is NAME else leaf_map[type_]
        if value in reserved_syntax_strings:
            transition = reserved_syntax_strings[value]
            if type_ is NAME:
                cls = Keyword

        tos = stack[-1]
        nodes = tos.nodes
        while True:
            dfa = tos.dfa
            try:
                plan = dfa.transitions[transition]
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
                    append(nodes, new_node)
                else:
                    self.error_recovery(token)
                    return
            except IndexError:
                raise InternalParseError("too much input", type_, value, start_pos)

        tos.dfa = plan.next_dfa
        for dfa in plan.dfa_pushes:
            node = StackNode()
            node.dfa = dfa
            node.nodes = []
            append(stack, node)
        leaf = cls()
        leaf.value = value
        leaf.start_pos = start_pos
        leaf.prefix = prefix
        append(stack[-1].nodes, leaf)

    Parser._add_token = _add_token


# Optimizes _defined_names to use heuristical order and return early based
# on the most likely node type. Also eliminates recursion.
def optimize_defined_names():
    from parso.python import tree

    append = list.append
    extend = list.extend
    pop = list.pop

    def _defined_names(current, include_setitem):
        # Return early.
        current_type = current.type
        if current_type not in {'testlist_star_expr', 'testlist_comp', 'exprlist', 'testlist', 'atom', 'star_expr', 'power', 'atom_expr'}:
            return [current]

        # Proceed to pool loop.
        elif current_type in {'testlist_star_expr', 'testlist_comp', 'exprlist', 'testlist', 'atom'}:
            pool = current.children[::2]
        # Proceed to pool loop.
        elif current_type in {'atom', 'star_expr'}:
            pool = [current.children[1]]

        # Return
        else:
            names = []
            trailer_children = current.children[-1].children
            value = trailer_children[0].value
            if value == ".":  # String comparisons use direct check, skip overload.
                append(names, trailer_children[1])
            elif value == '[' and include_setitem:
                for node in current.children[-2::-1]:
                    node_type = node.type
                    if node_type == 'trailer':
                        append(names, node.children[1])
                        break
                    elif node_type == 'name':
                        append(names, node)
                        break
            return names

        # Pool loop
        names = []
        while pool:
            current = pop(pool)
            current_type = current.type
            if current_type not in {'testlist_star_expr', 'testlist_comp', 'exprlist', 'testlist', 'atom', 'star_expr', 'power', 'atom_expr'}:
                append(names, current)

            elif current_type == 'atom_expr' or current_type == "power":
                # if children[-2] != '**':  # XXX Never True. children[-2] doesn't implement __eq__ and isn't a string.
                trailer_children = current.children[-1].children
                value = trailer_children[0].value

                if value == ".":  # String comparisons use direct check, skip overload.
                    append(names, trailer_children[1])

                elif value == '[' and include_setitem:
                    for node in current.children[-2::-1]:
                        node_type = node.type
                        if node_type == 'trailer':
                            append(names, node.children[1])
                            break
                        elif node_type == 'name':
                            append(names, node)
                            break

            elif current_type in {'testlist_star_expr', 'testlist_comp', 'exprlist', 'testlist'}:
                extend(pool, current.children[::2])
            else:  # {'atom', 'star_expr'}
                append(pool, current.children[1])
        return names

    _deferred_patch_function(tree._defined_names, _defined_names)


# Optimizes ExprStmt.get_defined_names to inline the popular heuristic.
def optimize_get_defined_names():
    from parso.python.tree import ExprStmt, _defined_names

    def get_defined_names(self, include_setitem=False):
        names = []
        append = names.append
        extend = names.extend
        children = self.children

        if children[1].type == 'annassign':
            names = _defined_names(children[0], include_setitem)

        i = 1
        for child in children[1:len(children) - 1:2]:
            if "=" in child.value:
                child = children[i - 1]
                if child.type not in {'testlist_star_expr', 'testlist_comp', 'exprlist', 'testlist', 'atom', 'star_expr', 'power', 'atom_expr'}:
                    append(child)
                else:
                    extend(_defined_names(child, include_setitem))
            i += 2
        return names
    ExprStmt.get_defined_names = get_defined_names


def optimize_check_flows():
    from jedi.inference.filters import ParserTreeFilter
    from jedi.inference.flow_analysis import REACHABLE, UNREACHABLE, reachability_check
    from operator import attrgetter
    cache = _register_cache()
    append = list.append
    key = attrgetter("line", "column")

    def _check_flows(self, names):
        try: return cache[names]
        except:  # Assume KeyError, but skip the lookup.
            ret = []
            context = self._node_context
            value_scope = self._parser_scope
            origin_scope = self._origin_scope
            for name in sorted(names, key=key, reverse=True):
                check = reachability_check(
                    context=context,
                    value_scope=value_scope,
                    node=name,
                    origin_scope=origin_scope)
                if check is not UNREACHABLE: append(ret, name)
                if check is REACHABLE: break
            cache[names] = ret
            return ret
    ParserTreeFilter._check_flows = _check_flows


def optimize_parsertreefilter_filter():
    """
    Replaces 'ParserTreeFilter._filter' and inlines '_is_name_reachable' and 'get_cached_parent_scope'.
    """
    from jedi.inference.filters import ParserTreeFilter
    from jedi.parser_utils import get_cached_parent_scope, get_parent_scope

    super_filter = _get_super_meth(ParserTreeFilter, "_filter")

    # Cache is taken from the optimized version
    cache = get_cached_parent_scope.__closure__[0].cell_contents
    assert type(cache) is dict, f"parent scope cache was never overridden, cache was {cache}"
    append_meth = list.append

    """

class StubFilter(ParserTreeFilter):
    name_class = StubName

    def _is_name_reachable(self, name):
        if not super()._is_name_reachable(name):
            return False

        # Imports in stub files are only public if they have an "as"
        # export.
        definition = name.get_definition()
        if definition.type in ('import_from', 'import_name'):
            if name.parent.type not in ('import_as_name', 'dotted_as_name'):
                return False
        n = name.value
        # TODO rewrite direct return
        if n.startswith('_') and not (n.startswith('__') and n.endswith('__')):
            return False
        return True
    """
    from jedi.inference.gradual.stub_value import StubFilter
    is_stub_filter = StubFilter.__instancecheck__

    cache2 = _register_cache()
    flow_cache = _register_cache()
    stub_cache = _register_cache()

    stub_filters = {}

    import builtins
    builtin_keys = set(k for k in builtins.__dict__.keys() if not k.startswith("_"))

    def _filter(self, names):

        parso_cache_node = self._parso_cache_node
        try:
            cache_node = cache[parso_cache_node]
        except:
            cache_node = cache[parso_cache_node] = {}

        key = self._until_position, names
        try:
            names = cache2[key]
        except:
            names = cache2[key] = (super_filter(self, names))

        is_stub = is_stub_filter(self)
        parser_scope = self._parser_scope
        ret = []
        for name in names:
            parent = name.parent
            ptype = parent.type
            if ptype != "trailer":
                base_node = parent if ptype in {"classdef", "funcdef"} else name
                # TODO: Find out if not having this breaks anything.
                # XXX: This doesn't seem to trigger? Remove line?
                # XXX: Seems to trigger on completions inside functions
                # if self._parso_cache_node is None:
                #     scope = get_parent_scope(base_node, False)
                # else:
                try:
                    scope = cache_node[base_node]
                except:  # Assume KeyError
                    scope = cache_node[base_node] = get_parent_scope(base_node, False)

                if scope is parser_scope:
                    # StubFilter-specific. Removes imports in .pyi
                    if is_stub:
                        modname = self._node_context._value.string_names[0]
                        try:
                            sf = stub_filters[modname]
                        except:
                            try:
                                mod = __import__(modname)
                            except:
                                sf = stub_filters[modname] = set()
                            else:
                                keys = []
                                for k in mod.__dict__.keys():
                                    keys.append(k)
                                sf = stub_filters[modname] = set(keys)

                        try:
                            definition = stub_cache[name]
                        except:
                            definition = stub_cache[name] = name.get_definition()

                        if definition.type in {'import_from', 'import_name'}:
                            if ptype not in {'import_as_name', 'dotted_as_name'}:
                                continue
                        n = name.value
                        if n not in sf:
                            continue
                        if n == "ellipsis":
                            continue  # Not sure why builtins even include this

                        if n[0] is "_":
                            try:
                                if n[1]  is not "_" or n[-1] is not "_" is not n[-2]:
                                    continue
                            except:
                                continue
                            # if n.startswith('_') and not (n.startswith('__') and n.endswith('__')):
                            #     continue
                    append_meth(ret, name)
        t = tuple(ret)
        try:
            ret = flow_cache[t]
            return ret
        except:
            ret = flow_cache[t] = self._check_flows(t)
            return ret

        # # NOTE: Don't inline _check_flows - SelfAttributeFilter overrides it.
        # return self._check_flows(tuple(ret))

    ParserTreeFilter._filter = _filter


# Stub modules are generally files. They should be cached and kept
# in memory for the entirety of the app lifetime. 5x speedup.
def optimize_stubfilter():
    from jedi.inference.gradual.stub_value import StubFilter
    stub_values_cache = {}
    values_meth = StubFilter.values

    def stub_values(self: StubFilter):
        try:
            return stub_values_cache[self.parent_context]
        except:
            stub_values_cache[self.parent_context] = values_meth(self)
            return stub_values_cache[self.parent_context]
    StubFilter.values = stub_values


def optimize_abstractusednamesfilter_values():
    from jedi.inference.filters import _AbstractUsedNamesFilter
    from jedi.inference import filters
    from parso.python.tree import Name

    # Assumes 'optimize_get_definition_names' was called first.
    cache = filters._get_definition_names.__closure__[0].cell_contents
    append = list.append
    extend = list.extend

    def values(self: _AbstractUsedNamesFilter):
        _filter = self._filter
        parso_cache_node = self._parso_cache_node
        used_names = self._used_names
        ret = []

        if parso_cache_node is not None:
            try:
                node_cache = cache[parso_cache_node]
            except:  # Assume KeyError
                node_cache = cache[parso_cache_node] = {}

            # XXX: Copy of optimized _get_definition_names, but with node cache outside loop
            for name_key in used_names:
                try:
                    tmp = node_cache[name_key]
                except:  # Assume KeyError
                    tmp = tuple(n for n in used_names[name_key] if n.is_definition(include_setitem=True))
                    node_cache[name_key] = tmp
                extend(ret, _filter(tmp))
        else:
            for name_key in used_names:
                tmp = tuple(n for n in used_names[name_key] if n.is_definition(include_setitem=True))
                extend(ret, _filter(tmp))
        return self._convert_names(ret)
    _AbstractUsedNamesFilter.values = values


def optimize_get_cached_parent_scope():
    cache = _register_cache()
    from jedi import parser_utils
    from jedi.parser_utils import get_parent_scope

    def get_cached_parent_scope(parso_cache_node, node, include_flows=False):
        if parso_cache_node is None:  # TODO: Find out if not having this breaks anything.
            return get_parent_scope(node, include_flows)
        try:
            return cache[parso_cache_node][node]
        except:  # Assume KeyError
            if parso_cache_node not in cache:
                cache[parso_cache_node] = {}
            ret = cache[parso_cache_node][node] = get_parent_scope(node, include_flows)
            return ret

    parser_utils.get_cached_parent_scope = get_cached_parent_scope


def optimize_get_parent_scope():
    from parso.python import tree

    is_flow = tree.Flow.__instancecheck__
    cache = _register_cache()
    def get_parent_scope(node, include_flows=False):
        if node in cache:
            return cache[node]

        scope = node.parent

        if scope is not None:

            tmp = node
            try:
                while True:tmp = tmp.children[0]
            except:
                node_start_pos = tmp.line, tmp.column

            node_parent = node.parent
            parent_type = node_parent.type

            if (cont := not (parent_type == 'param' and node_parent.name == node)):
                if (cont := not (parent_type == 'tfpdef' and node_parent.children[0] == node)):
                    pass

            while True:
                if ((t := scope.type) == 'comp_for' and scope.children[1].type != 'sync_comp_for') or \
                        t in {'file_input', 'classdef', 'funcdef', 'lambdef', 'sync_comp_for'}:
                    if scope.type in {'classdef', 'funcdef', 'lambdef'}:

                        for child in scope.children:
                            try:
                                if child.value is ":": break
                            except: continue
                        if (child.line, child.column) >= node_start_pos:
                            if cont:
                                scope = scope.parent
                                continue
                    cache[node] = scope
                    return scope
                elif include_flows and is_flow(scope):
                    if t == 'if_stmt':
                        for n in scope.get_test_nodes():
                            tmp = n
                            try:
                                while True: tmp = tmp.children[0]
                            except:start = tmp.line, tmp.column
                            if start <= node_start_pos:
                                tmp = n
                                try:
                                    while True: tmp = tmp.children[-1]
                                except:end = tmp.end_pos
                                if node_start_pos < end:
                                    break
                        else:
                            cache[node] = scope
                            return scope
                scope = scope.parent
        cache[node] = None
        return None  # It's a module already.

    from jedi import parser_utils
    _deferred_patch_function(parser_utils.get_parent_scope, get_parent_scope)


# Patches inspect.getmodulename to use heuristics.
def optimize_getmodulename():
    module_suffixes = {suf: -len(suf) for suf in sorted(
        __import__("importlib").machinery.all_suffixes() + [".pyi"], key=len, reverse=True)}
    suffixes = tuple(module_suffixes)
    module_suffixes = module_suffixes.items()
    rpartition = str.rpartition
    import re
    match = re.compile(rf"^.*\.({'|'.join(s[1:] for s in suffixes)})$").match

    def getmodulename(name: str):
        if match(name):
            for suf, neg_suflen in module_suffixes:
                if name[neg_suflen:] == suf:
                    return rpartition(name, "\\")[-1][:neg_suflen]
        return None
    import inspect
    _deferred_patch_function(inspect.getmodulename, getmodulename)


# Improve the performance of platform.system() for win32.
def optimize_platform_system():
    import platform
    import sys
    if sys.platform == "win32":
        platform._system = platform.system  # Backup
        platform.system = "Windows".__str__


def optimize_name_get_definition():
    # tot 32.05, cum 186.302 get_definition 717
    from parso.python.tree import Name, _GET_DEFINITION_TYPES
    from dev_utils import ncalls
    def get_definition(self, import_name_always=False, include_setitem=False):
        node = self.parent
        node_type = node.type
        if node_type in {'funcdef', 'classdef'}:
            if self == node.name:
                return node
        elif node_type == 'except_clause':
            if self.get_previous_sibling() == 'as':
                return node.parent
        while node is not None:
            if node.type in _GET_DEFINITION_TYPES:
                if self in node.get_defined_names(include_setitem):
                    return node
                break
            node = node.parent
        return None

    Name.get_definition = get_definition


def optimize_get_used_names():
    """Replaces parso.python.tree.Module.get_used_names"""

    append = list.append
    pop = list.pop

    def get_used_names(self):
        if used_names := self._used_names:
            return used_names
        pool = [(self,)]
        self._used_names = used_names = {}
        while pool:
            for node in pop(pool):
                type = node.type
                if type in {()}:
                    pass
                elif type == "name":
                    try:
                        append(used_names[node.value], node)
                    except:
                        used_names[node.value] = [node]
                else:
                    try:
                        append(pool, node.children)
                    except:
                        patch_terminals(type)
        return used_names

    def patch_terminals(*terminal_types):
        consts = list(get_used_names.__code__.co_consts)
        for i, c in enumerate(consts):
            if isinstance(c, frozenset):
                consts[i] = frozenset((*terminal_types, *c))
                get_used_names.__code__ = get_used_names.__code__.replace(
                    co_consts=tuple(consts))
                return

    patch_terminals(
        'newline', 'string', 'number', 'fstring_string', 'fstring_start',
        'operator', 'fstring_end', 'endmarker', 'keyword')

    from parso.python.tree import Module
    Module.get_used_names = get_used_names


def optimize_valuewrapperbase_name():
    from jedi.inference.names import ValueName
    from jedi.inference.compiled import CompiledValueName
    from jedi.inference.utils import UncaughtAttributeError
    from jedi.inference import base_value

    def name(self):
        try:
            if name := self._wrapped_value.name.tree_name:
                return ValueName(self, name)
            return CompiledValueName(self, name)
        except AttributeError as e:
            raise UncaughtAttributeError(e) from e
    base_value._ValueWrapperBase.name = property(name)


def optimize_get_definition_names():
    """Optimizes _get_definition_names in 'jedi/inference/filters.py'"""
    cache = _register_cache()
    def _get_definition_names(parso_cache_node, used_names, name_key):
        try:
            return cache[parso_cache_node][name_key]
        except:  # Assume KeyError, but skip the error lookup
            try:
                ret = tuple(n for n in used_names[name_key] if n.is_definition(include_setitem=True))
            except:  # Unlikely. Assume KeyError, but skip the error lookup
                ret = ()
            if parso_cache_node is not None:
                try:
                    cache[parso_cache_node][name_key] = ret
                except:  # Unlikely. Assume KeyError, but skip the error lookup
                    cache[parso_cache_node] = {name_key: ret}
            return ret

    from jedi.inference import filters
    filters._get_definition_names = _get_definition_names


def optimize_classfilter_filter():
    """Replaces 'ClassFilter._filter' and inlines '_access_possible'."""
    from jedi.inference.value.klass import ClassFilter
    super_filter = _get_super_meth(ClassFilter, "_filter")
    startswith = str.startswith
    endswith = str.endswith
    append = list.append

    def _filter(self, names):
        ret = []
        _is_instance = self._is_instance
        _equals_origin_scope = self._equals_origin_scope
        for name in super_filter(self, names):
            if not _is_instance:
                try:  # Most likely
                    if (expr_stmt := name.get_definition()).type == 'expr_stmt':
                        if (annassign := expr_stmt.children[1]).type == 'annassign':
                            # If there is an =, the variable is obviously also
                            # defined on the class.
                            if 'ClassVar' not in annassign.children[1].get_code() and '=' not in annassign.children:
                                continue
                except:  # Assume AttributeError
                    pass
            if not startswith(v := name.value, "__") or endswith(v, "__") or _equals_origin_scope():
                append(ret, name)
        return ret
    ClassFilter._filter = _filter


def optimize_selfattributefilter_filter():
    """
    Replaces 'SelfAttributeFilter._filter' and inlines '_filter_self_names'.
    """
    from jedi.inference.value.instance import SelfAttributeFilter
    from operator import attrgetter
    cache = _register_cache()  # XXX: Cache for start and end descriptors
    append = list.append
    get_ends = attrgetter("start_pos", "end_pos")
    cache2 = _register_cache()
    def _filter(self, names):
        try: start, end = cache[(scope := self._parser_scope)]
        except: start, end = cache[scope] = get_ends(scope)
        ret = []
        for name in names:
            if name in cache2:
                if cache2[name]:
                    append(ret, name)
            else:
                if start < (name.line, name.column) < end:
                    if (trailer := name.parent).type == 'trailer':
                        children = trailer.parent.children
                        if len(children) is 2 and trailer.children[0] is '.':
                            if name.is_definition() and self._access_possible(name):
                                if self._is_in_right_scope(children[0], name):
                                    cache2[name] = True
                                    append(ret, name)
                                    continue
                cache2[name] = False
        return ret
    SelfAttributeFilter._filter = _filter


def optimize_split_lines():
    splitlines = str.splitlines
    append = list.append

    def split_lines(string: str, keepends: bool = False) -> list[str]:
        lines = splitlines(string, keepends)
        try:
            if string[-1] is "\n":
                append(lines, "")
        except:  # Assume IndexError, but skip the lookup
            append(lines, "")
        return lines

    from parso import utils
    _deferred_patch_function(utils.split_lines, split_lines)


def optimize_fuzzy_match():
    """Optimizes '_fuzzy_match' in jedi.api.helpers."""
    _index = str.index
    def _fuzzy_match(string: str, like_name: str):
        index = -1
        try:
            for char in like_name:
                index = _index(string, char, index + 1)
                continue
        except:  # Expect IndexError but don't bother looking up the name.
            return False
        return True
    from jedi.api import helpers
    helpers._fuzzy_match = _fuzzy_match


def optimize_filter_names():
    from jedi.api.classes import Completion
    from jedi.api.helpers import _fuzzy_match
    from jedi.api import completion
    class NewCompletion(Completion):
        __init__ = None.__init__
        _is_fuzzy = False

    class FuzzyCompletion(Completion):
        __init__ = None.__init__
        _is_fuzzy = True
        complete = property(None.__init__)

    _len = len
    fuzzy_match = _fuzzy_match
    startswith = str.startswith
    append = list.append
    lower = str.lower
    strip = str.__str__

    def filter_names(inference_state, completion_names, stack, like_name, fuzzy, cached_name):
        like_name_len = _len(like_name)
        if fuzzy:
            match_func = fuzzy_match
            completion_base = FuzzyCompletion
            do_complete = None.__init__         # Dummy
        else:
            match_func = startswith
            completion_base = NewCompletion
            # if settings.add_bracket_after_function:
            #     def do_complete():
            #         if new.type == "function":
            #             return n[like_name_len:] + "("
            #         return n[like_name_len:]
            # else:
            #     def do_complete():
            #         return n[like_name_len:]

        class completion(completion_base):
            _inference_state = inference_state  # InferenceState object is global.
            _like_name_length = like_name_len   # Static
            _cached_name = cached_name          # Static
            _stack = stack                      # Static


        if settings.case_insensitive_completion:
            case = lower
            like_name = case(like_name)
        else:
            case = strip  # Dummy
        ret = []
        dct = {}

        if not like_name:
            for e in completion_names:
                n_ = e.string_name
                if n_ not in dct:
                    dct[n_] = None
                    if ((tn := e.tree_name) is None) or (getattr(tn.get_definition(), "type", None) != "del_stmt"):
                        new = completion()
                        new._name = e
                        append(ret, new)

        else:
            for e in completion_names:
                n_ = e.string_name
                if case(n_[0]) == like_name[0]:
                    if case(n_[:like_name_len]) == like_name:
                        # Store unmodified so names like Cache and cache aren't merged.
                        if (k := (n_, n_[like_name_len:])) not in dct:
                            dct[k] = None
                            if ((tn := e.tree_name) is None) or (getattr(tn.get_definition(), "type", None) != "del_stmt"):
                                new = completion()
                                new._name = e
                                append(ret, new)
        return ret
    completion.filter_names = filter_names


def optimize_valuewrapperbase_getattr():
    """Replaces _ValueWrapperBase.__getattr__.
    __getattr__ works like dict's __missing__. When an attribute error is
    thrown, use this opportunity to set a cached value.
    """
    _getattr = getattr
    _setattr = setattr
    def __getattr__(self, name):
        ret = _getattr(self._wrapped_value, name)
        _setattr(self, name, ret)
        return ret
    from jedi.inference.base_value import _ValueWrapperBase
    _ValueWrapperBase.__getattr__ = __getattr__


def optimize_lazyvaluewrapper_wrapped_value():
    def _wrapped_value(self):
        if "_cached_value" in (d := self.__dict__):
            return d["_cached_value"]
        result = d["_cached_value"] = self._get_wrapped_value()
        return result

    from jedi.inference.base_value import LazyValueWrapper
    LazyValueWrapper._wrapped_value = property(_wrapped_value)
