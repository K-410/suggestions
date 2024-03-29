"""This module implements classes extending Jedi's inference capability."""

from jedi.inference.compiled.value import (
    CompiledValue, CompiledValueFilter, CompiledName, CompiledModule)
from jedi.inference.value.instance import SelfName, CompiledBoundMethod
from jedi.inference.value.instance import BoundMethod
from jedi.inference.value.function import FunctionValue, MethodValue
from jedi.inference.syntax_tree import tree_name_to_values
from jedi.inference.value.klass import ClassValue, ClassName
from jedi.inference.lazy_value import LazyKnownValues, LazyTreeValue
from jedi.inference.base_value import ValueWrapper
from jedi.inference.arguments import AbstractArguments, TreeArguments
from jedi.inference.signature import TreeSignature
from jedi.inference.compiled import builtin_from_name
from jedi.inference.context import CompiledContext, GlobalNameFilter
from jedi.inference.imports import Importer
from jedi.inference.param import ExecutedParamName
from jedi.api.interpreter import MixedModuleContext, MixedParserTreeFilter
from jedi.inference.names import TreeNameDefinition, NO_VALUES, ValueSet, StubName
from jedi.inference.value import CompiledInstance, ModuleValue

from jedi.file_io import FileIOFolderMixin
from jedi.api import Script, Completion

from parso.python.tree import Name, ClassOrFunc, Lambda
from parso.file_io import KnownContentFileIO
from parso.tree import search_ancestor, BaseNode

from itertools import repeat, compress, count
from operator import attrgetter, methodcaller
from pathlib import Path
import types

from functools import partial

from textension.utils import (
    _forwarder, _named_index, _unbound_getter, consume, falsy_noargs, inline,
     set_name, truthy_noargs, instanced_default_cache, Aggregation, starchain,
     lazy_overwrite, namespace, filtertrue, get_mro_dict, Variadic, _variadic_index)

from .tools import state, get_handle, ensure_blank_eol
from .. import settings

import bpy


_closures = []

# For VirtualModule overrides. Assumes patch_Importer_follow was called.
Importer_redirects = {}
CompiledModule_redirects = {}

memoize_values = state.memoize_cache.values()
get_start_pos = attrgetter("line", "column")
get_cursor_focus = attrgetter("select_end_line_index", "select_end_character")

runtime = namespace(is_reset=True)

# Used by various optimizations.
for cls in iter(node_types := [BaseNode]):
    node_types += cls.__subclasses__()
node_types: frozenset[BaseNode] = frozenset(node_types)


def add_module_redirect(virtual_module, name=None):
    if name is None:
        string_names = getattr(virtual_module, "string_names", None)
        assert isinstance(string_names, tuple), (string_names, virtual_module)
        name = ".".join(string_names)
    Importer_redirects[name] = virtual_module


@inline
def is_namenode(node) -> bool:
    return Name.__instancecheck__

@inline
def filter_names(sequence):
    return partial(filter, is_namenode)

@inline
def is_leaf(node) -> bool:
    from parso.tree import Leaf
    return Leaf.__instancecheck__

@inline
def is_classdef(node) -> bool:
    from parso.python.tree import Class
    return Class.__instancecheck__

@inline
def is_str(obj) -> bool:
    return str.__instancecheck__

@inline
def filter_strings(seq):
    return partial(filter, str.__instancecheck__)

@inline
def get_type_name(cls: type) -> str:
    return type.__dict__["__name__"].__get__

@inline
def get_fget(property_obj):
    return property.__dict__["fget"].__get__

@inline
def _rebuild_closure_caches():
    return partial(map, partial.__call__, _closures, map(dict.__new__, repeat(dict)))

@inline
def map_dict_clear(dicts):
    return partial(map, dict.clear)

@inline
def get_memoize_dicts():
    return partial(filtertrue, memoize_values)

@inline
def is_funcdef(node) -> bool:
    from parso.python.tree import Function
    return Function.__instancecheck__

@inline
def filter_funcdefs(nodes):
    return partial(filter, is_funcdef)

@inline
def is_basenode(node) -> bool:
    from parso.tree import BaseNode
    return BaseNode.__instancecheck__

@inline
def filter_basenodes(nodes):
    return partial(filter, is_basenode)

@inline
def is_param(node) -> bool:
    from parso.python.tree import Param
    return Param.__instancecheck__

@inline
def filter_params(nodes):
    return partial(filter, is_param)

@inline
def is_pynode(node) -> bool:
    from parso.python.tree import PythonNode
    return PythonNode.__instancecheck__

@inline
def filter_pynodes(nodes):
    return partial(filter, is_pynode)

@inline
def is_operator(node) -> bool:
    from parso.python.tree import Operator
    return Operator.__instancecheck__

@inline
def filter_operators(nodes):
    return partial(filter, is_operator)

@inline
def is_keyword(obj) -> bool:
    from parso.python.tree import Keyword
    return Keyword.__instancecheck__

@inline
def filter_keywords(nodes):
    return partial(filter, is_keyword)

@inline
def is_number(obj) -> bool:
    from parso.python.tree import Number
    return Number.__instancecheck__

@inline
def filter_numbers(nodes):
    return partial(filter, is_number)

@inline
def map_types(seq):
    return partial(map, attrgetter("type"))

@inline
def map_values(numbers):
    return partial(map, attrgetter("value"))

@inline
def map_infer(bases):
    return partial(map, methodcaller("infer"))

@inline
def map_eq(sequence1, sequence2) -> map:
    from operator import eq
    return partial(map, eq)

@inline
def map_startswith(iterable1, iterable2):
    return partial(map, str.startswith)


# Used by VirtualValue.py__call__ and other inference functions where we
# don't have any arguments to unpack. For constructors.
@inline
class NoArguments(AbstractArguments):
    def unpack(self, *args, **kw):
        yield from ()


@inline
def filter_node_type(node_type, seq):
    from textension.utils import instanced_default_cache
    from itertools import compress, repeat
    from functools import partial
    from operator import eq
    from .common import map_types

    @instanced_default_cache
    def node_types(self: dict, node_type):
        self[node_type] = partial(map, eq, repeat(node_type))
        return self[node_type]

    def filter_node_type(node_type, seq):
        return compress(seq, node_types[node_type](map_types(seq)))

    return filter_node_type


@inline
def create_node_type_filter(node_type: str):
    from textension.utils import close_cells
    from operator import eq

    cache = {}

    def create_node_type_filter(node_type):
        if node_type not in cache:
            cache[node_type] = partial(map, eq, repeat(node_type))

        mapper = cache[node_type]

        @close_cells(mapper)
        def filter_nodes(nodes):
            return compress(nodes, mapper(map_types(nodes)))
        return filter_nodes

    return create_node_type_filter


@inline
def filter_suites(nodes):
    return create_node_type_filter("suite")


@inline
def filter_class_or_func(nodes):
    from parso.python.tree import ClassOrFunc
    return partial(filter, ClassOrFunc.__instancecheck__)


def yield_once(func):
    memo = state.memoize_cache[func] = {}
    def wrapper(*key):
        if key in memo:
            return
        memo[key] = result = list(func(*key))
        yield from result
    return wrapper


@instanced_default_cache
def sessions(self: dict, text_id):
    file_io = BpyTextBlockIO(Path(f"Text({text_id})"))
    module = BpyTextModule(file_io)
    return self.setdefault(text_id, TextSession(text_id, module))


def register_cache(func):
    def inner(wrap):
        func_name = getattr(func, "__name__", None) or str(func)
        assert isinstance(wrap.__closure__[0].cell_contents, dict)
        _closures.append(partial(setattr, wrap.__closure__[0], "cell_contents"))
        return set_name(func_name + " (cached)")(wrap)
    return inner


def reset_state():
    """Reset the global inference state, releasing objects created during
    completion, except from things like stub modules whose lifetime should
    persist for the duration of Blender.
    """

    # Assign a new cache to registered closures.
    # This is for functions that use the optimized state cache.
    consume(_rebuild_closure_caches())

    # Clear the cache for functions that use the stock memoize.
    consume(map_dict_clear(get_memoize_dicts()))

    # Persistent state means we have to clear access handles or they'll cause
    # circular references undetectable by the garbage collector and leak.
    state._access_handles.clear()

    state.reset_recursion_limitations()
    state.inferred_element_counts = {}
    runtime.is_reset = True


@inline
def add_pending_state_reset():
    return partial(bpy.app.timers.register, reset_state)


def state_cache(func):
    cache = {}
    @register_cache(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper


def state_cache_default(default):
    """Inference state cache with recursion mitigation."""

    def decorator(func):
        cache = {}
        @register_cache(func)
        def wrapper(*args):
            if args not in cache:
                cache[args] = default
                cache[args] = func(*args)
            return cache[args]
        return wrapper
    return decorator


def state_cache_kw(func):
    """Same as state_cache, but for functions using keyword arguments."""
    from textension.utils import dict_items
    from builtins import tuple

    cache = {}
    @register_cache(func)
    def wrapper(*args, **kw):
        key = (args, tuple(dict_items(kw)))
        if key not in cache:
            cache[key] = func(*args, **kw)
        return cache[key]
    return wrapper


# Does what ``Abstractfilter._filter()`` does, just a lot faster.
@inline
def filter_until(pos: int | None, names):
    from itertools import compress
    from builtins import map

    @inline
    def map_start_pos(names):
        return partial(map, get_start_pos)

    def filter_until(pos: int | None, names):
        if pos:
            return compress(names, map(pos.__gt__, map_start_pos(names)))
        return names

    return filter_until


# Meant to be called by GlobalNameFilter.get
@inline
def get_global_statements(module):
    from parso.python.tree import GlobalStmt, PythonMixin
    from textension.utils import defaultdict_list
    from .common import filter_suites

    # PythonNode and PythonBaseNode share this. Needed for filtering scopes and statements.
    filter_py_base_node = partial(filter, PythonMixin.__instancecheck__)
    @inline
    def map_children(nodes):
        return partial(map, attrgetter("children"))

    @state_cache
    def get_global_statements(module):
        globs = defaultdict_list()

        # In case of large modules, start with direct functions/classes.
        pool = [filter_class_or_func(module.children)]

        for node in filter_py_base_node(starchain(pool)):
            node_type = node.type

            if node_type == "simple_stmt":
                if node.children[0].__class__ is GlobalStmt:
                    name = node.children[0].children[1]
                    globs[name.value] += name,

            elif node_type in {"newline", "error_node", "error_leaf"}:
                continue

            elif node_type in {"classdef", "funcdef"}:
                # The suite. We're just looking for valid global statements
                # within a function.
                pool += node.children[-1].children,

            elif node_type == "decorated":
                pool += (node.children[1],),

            # Flow types.
            elif node_type in {"for_stmt", "if_stmt", "try_stmt", "while_stmt", "with_stmt"}:
                pool += map_children(filter_suites(node.children)),

        return globs
    return get_global_statements


@state_cache
def get_scope_name_definitions(scope):

    # Jedi allows ill-formed function/lambda constructs, so we cannot assume
    # ``scope`` is always a BaseNode. This is obviously a workaround for an
    # issue in paro's grammar, but I'm not going to touch that.
    try:
        pool = scope.children[:]
    except AttributeError:  # Not a BaseNode.
        return ()


    # XXX: This code is for testing.
    # def check(name):
    #     if not is_namenode(name):
    #         pass
    # class L(list):
    #     def __iadd__(self, it):
    #         check(it[0])
    #         return super().__iadd__(it)
    # namedefs = L()

    namedefs = []

    for n in filter_basenodes(pool):
        if n.type in {"classdef", "funcdef"}:
            # These are always definitions.
            namedefs += n.children[1],

        elif n.type == "simple_stmt":
            n = n.children[0]

            if n.type == "expr_stmt":
                name = n.children[0]

                # Could be ``atom_expr``, as in dotted name. Skip those.
                if name.type == "name":
                    namedefs += name,

                # ``a, b = X``. Here we take ``a`` and ``b``.
                # Could be ``atom_expr``, i.e ``x.a, y.a = Z``, skip those.
                elif name.type == "testlist_star_expr":
                    namedefs += filter_names(name.children[::2])

                # ``a.b = c``. Not a definition.
                elif name.type == "atom_expr":
                    pass

                # Could be anything. Use name.get_definition().
                elif name.type == "atom":
                    pool += name.children

                else:
                    print("get_scope_name_definitions unhandled:", name.type)

            # The only 2 import statement types.
            elif n.type in {"import_name", "import_from"}:
                name = n.children[-1]

                # ``import collections.abc`` is not a definition.
                if name.type == "dotted_name":
                    continue

                # The definition is potentially the last child, but
                # watch out for any closing parens or star operator.
                if name.type == "operator":
                    if name.value == "*":
                        continue
                    name = n.children[-2]

                if name.type in {"import_as_names", "dotted_as_names"}:
                    for name in name.children[::2]:
                        if name.type != "name":
                            name = name.children[2]
                        namedefs += name,

                else:
                    if name.type in {"import_as_name", "dotted_as_name"}:
                        name = name.children[2]
                    namedefs += name,

            # ``atom`` nodes are too ambiguous to extract names from.
            # Just into the pool and look for names the old way.
            elif n.type == "atom":
                pool += n.children

            elif n.type == "del_stmt":
                name = n.children[1]
                if name.type == "name":
                    selectors = map_eq(repeat(name.value), map_values(namedefs))
                else:
                    del_names = set(map_values(filter_names(n.children[1].children)))
                    selectors = map(del_names.__contains__, map_values(namedefs))
                for index in reversed(list(compress(count(), selectors))):
                    del namedefs[index]

        elif n.type == "decorated":
            pool += n.children[1],

        elif n.type in {"for_stmt", "if_stmt"}:
            # Usually the suite, but could be a Name.
            # Example: ``for x in y: x|<--`` is a Name.
            suite = n.children[-1]
            if suite.type == "suite":
                pool += n.children[-1].children

            name = n.children[1]
            if name.type == "comparison":
                name = name.children[0]

            if name.type == "exprlist":
                namedefs += filter_names(name.children)
            elif name.type == "name":
                namedefs += name,

        elif n.type in {"error_node", "arglist"}:
            continue

        elif n.type == "try_stmt":
            for c in filter_basenodes(n.children):
                if c.type == "suite":
                    pool += c.children
                elif c.type == "except_clause":
                    for name in filter_names(c.children):
                        if name.get_previous_sibling() == "as":
                            namedefs += name,

        # Expression assignment here is to avoid duplicate else-clause.
        elif n.type == "async_stmt" and (n := n.children[1]).type == "funcdef":
            namedefs += n.children[1],

        else:
            pool += n.children

    # Get definitions for the rest.
    for n in filter_names(pool):
        if n.get_definition(include_setitem=True):
            namedefs += n,

    return namedefs


@inline
def find_definition(ref: Name):
    scope_types = {"classdef", "comp_for", "file_input", "funcdef",  "lambdef",
                   "sync_comp_for"}

    def find_definition(ref: Name):
        value = ref.value
        scope = ref.parent

        while scope.type not in scope_types:
            scope = scope.parent

        if scope.type in {"funcdef", "classdef", "except_clause"}:

            children = scope.children
            if value == children[1].value:  # Is the function/class name definition.
                return children[1]

            elif scope.type == "except_clause" and value == children[-1].value:
                return children[-1]

            # ``ref`` is probably an argument.
            else:
                scope = scope.parent

        start = ref.line, ref.column
        for name in get_cached_scope_definitions(scope)[ref.value]:
            if (name.line, name.column) < start:
                return name

        node = ref
        while node := node.parent:
            # Skip the dot operator.
            if node.type != "error_node":
                for name in filter_names(node.children):
                    if name.value == ref.value and name is not ref:
                        return name
        return None
    return find_definition


@inline
def get_cached_scope_definitions(scope) -> dict[str, list]:
    from textension.utils import defaultdict_list
    from .common import get_scope_name_definitions

    empty_factory = repeat(()).__next__

    @state_cache
    def get_cached_scope_definitions(scope):
        cache = defaultdict_list()
        scope_definitions = get_scope_name_definitions(scope)
        for namedef in scope_definitions:
            cache[namedef.value] += namedef,

        cache.default_factory = empty_factory
        return cache
    return get_cached_scope_definitions


# This version doesn't include flow checks nor read or write to cache.
def get_parent_scope_fast(node):
    if scope := node.parent:
        pt = scope.type

        if either := pt in {"tfpdef", "param"}:
            either = (pt == "tfpdef" and scope.children[0] == node) or \
                     (pt == "param"  and scope.name == node)

        while True:
            stype = scope.type
            if stype in {"classdef", "funcdef", "lambdef", "file_input", "sync_comp_for", "comp_for"}:

                if stype in {"file_input", "sync_comp_for", "comp_for"}:
                    if stype != "comp_for" or scope.children[1].type != "sync_comp_for":
                        break

                elif either or get_start_pos(scope.children[-2]) < node.start_pos:
                    break
            scope = scope.parent
    return scope


@inline
def get_fget(property_obj):
    return property.__dict__["fget"].__get__


def infer_descriptor(property_obj, context):

    # Defensive programming here would be unreadable.
    try:
        func = get_fget(property_obj)
        import_names = tuple(func.__module__.split("."))
        module, = Importer(state, import_names, context).follow()

        # Jedi uses 1-based.
        line = func.__code__.co_firstlineno + 1
        for name in module.tree_node.get_used_names().get(func.__name__, ()):
            if name.line != line:
                continue

            context = module.as_context()
            cls_name = search_ancestor(name, "classdef").children[1]
            cls_val, = tree_name_to_values(state, context, cls_name)
            func_val, = tree_name_to_values(state, context, name)

            instance, = cls_val.execute(NoArguments)
            descriptor_val, = func_val.py__get__(instance, None)
            return descriptor_val
    except:
        pass
    return None


@inline
def get_submodule_names():
    from os import listdir
    from sys import modules
    from importlib.util import find_spec
    from jedi.inference.names import SubModuleName
    from textension.utils import get_module_dict, get_module_dir
    from itertools import repeat
    from types import ModuleType

    from importlib.machinery import all_suffixes
    from itertools import compress
    from builtins import map, len

    suffixes = (tuple(all_suffixes()),)
    endswith = str.endswith
    index = str.index

    def get_submodule_names(self, only_modules=True):
        result = []
        name = self.py__name__()

        # Exclude these as we already know they don't have sub-modules.
        if name not in {"builtins", "typing"}:
            m = self.obj
            assert m.__class__ is ModuleType

            exports = set()
            # It's possible we're passing non-module objects.
            # if isinstance(m, ModuleType):
            md = get_module_dict(m)
            exports.update(get_module_dir(m) + list(md))

            try:
                # ``__all__`` could be anything.
                exports.update(md.get("__all__", ()))
            except TypeError:
                pass

            try:
                # ``__path__`` could be anything.
                for path in md.get("__path__", ()):
                    names = listdir(path)
                    for name in compress(names, map(endswith, names, suffixes * len(names))):
                        exports.add(name[:index(name, ".")])
            except (TypeError, FileNotFoundError):
                pass

            names = []
            for e in filter_strings(exports):
                # Skip double-underscored names.
                if e[:2] == "__" == e[-2:]:
                    continue
                if only_modules:
                    full_name = f"{name}.{e}"
                    if full_name not in modules:
                        try:
                            assert find_spec(full_name)
                        except (ValueError, AssertionError, ModuleNotFoundError):
                            continue
                names += e,

            result = list(map(SubModuleName, repeat(self.as_context()), names))
        return result
    return get_submodule_names


@inline
def walk_scopes(scope, include_scopes: bool):
    from itertools import compress, count
    from parso.python.tree import Keyword

    flows  = {"for_stmt", "if_stmt", "try_stmt", "while_stmt", "with_stmt"}
    scopes = {"classdef", "comp_for", "file_input", "funcdef",  "lambdef",
              "sync_comp_for"}

    any_scope = scopes | flows
    is_keyword = Keyword.__instancecheck__

    @state_cache
    def walk_scopes(node, include_scopes):
        if include_scopes and (parent := node.parent):
            type = parent.type

            if is_param := type in {"tfpdef", "param"}:
                is_param = (type == 'tfpdef' and parent.children[0] == node) or \
                           (type == 'param'  and parent.name == node)

        scope = node
        while scope := scope.parent:
            t = scope.type
            if t in any_scope:
                if t in scopes:
                    if not include_scopes:
                        break
                    if t in {"classdef", "funcdef", "lambdef"}:
                        if not is_param and scope.children[-2].start_pos >= node.start_pos:  # type: ignore
                            continue
                    elif t == "comp_for" and scope.children[1].type != "sync_comp_for":
                        continue

                elif t == "if_stmt":
                    children = scope.children
                    for index in compress(count(), map(is_keyword, children)):
                        if children[index].value != "else":
                            n = children[index + 1]
                            start = node.start_pos
                            if start >= n.start_pos and start < n.end_pos:
                                break
                    else:
                        yield scope
                    continue
                yield scope
        return None
    return walk_scopes


# A modified version of reachability check used by various optimizations.
@inline
def trace_flow(node, origin_scope):
    from jedi.inference.flow_analysis import get_flow_branch_keyword, _break_check

    from .common import walk_scopes

    @state_cache
    def trace_flow(name, origin_scope):
        if origin_scope is not None:
            branch_matches = True
            for flow_scope in walk_scopes(origin_scope, False):  # 0.5ms (274)
                if flow_scope in walk_scopes(name, False):
                    node_keyword   = get_flow_branch_keyword(flow_scope, name)
                    origin_keyword = get_flow_branch_keyword(flow_scope, origin_scope)

                    if branch_matches := node_keyword == origin_keyword:
                        break

                    elif flow_scope.type == "if_stmt" or \
                        (flow_scope.type == "try_stmt" and origin_keyword == 'else' and node_keyword == 'except'):
                            return False

            if branch_matches:
                first_flow_scope = None
                for first_flow_scope in walk_scopes(name, True):
                    break
                while origin_scope:
                    if first_flow_scope is origin_scope:
                        return name
                    origin_scope = origin_scope.parent

        # XXX: For testing. What's the point of break check?
        # if first_flow_scope is None:
        #     first_flow_scope = next(walk_scope(name, True), None)
        # return _break_check(context, value_scope, first_flow_scope, name)
        return name
    return trace_flow


# This version differs from the stock ``_check_flows`` by:
# - Doesn't break after first hit. The stock version is designed to process
#   like-named sequences. Here we process *all* names in one go.
# - No pre-sorting. It's not applicable unless we break out of the loop.
def _check_flows(self, names):
    # if origin_scope := self._origin_scope:
    #     return filter(None, map(trace_flow, names, repeat(origin_scope)))
    # If origin scope is None, there's nothing to trace.
    return names


def _repr(obj):
    return type.__dict__['__name__'].__get__(type(obj))


def annotate(obj):
    if not isinstance(obj, type):
        obj = type(obj)

    import inspect
    ann = inspect.formatannotation(obj)
    if "bpy_types" in ann:
        ann = ann.replace("bpy_types", "bpy.types")
    return ann


def get_class_or_instance_type(name) -> str:
    try:
        value, = name.infer()
    except:
        return name.api_type

    if isinstance(value, CompiledBoundMethod):
        return "method"

    elif isinstance(value, VirtualValue):
        obj = value.obj

    elif isinstance(value, VirtualInstance):
        obj = value.class_value.obj

    else:
        if isinstance(value, ValueWrapper):
            value = value._wrapped_value

        if isinstance(value, CompiledValue):
            obj = value.access_handle.access._obj

        elif isinstance(value, FunctionValue):
            if isinstance(value, MethodValue):
                return "method"
            return "function"
        else:
            return value.py__name__()

    return annotate(obj)


def get_statement_type(name):
    for value in name.infer():
        if node := value.tree_node:
            if isinstance(node, ClassOrFunc):
                try:
                    return node.name.value
                except AttributeError:
                    assert isinstance(node, Lambda)
                    return "<lambda>"
        return value.api_type
    return None


def api_type_from_name(name):
    if isinstance(name, TreeNameDefinition):
        if definition := name.tree_name.get_definition(import_name_always=True):
            # Jedi will just put "module" on any import statement, which isn't useful.
            if definition.type in {"import_name", "import_from"}:
                for value in name.infer():
                    return value.api_type
            return name._API_TYPES.get(definition.type, "statement")
    return name.api_type


def get_extended_type(name):
    api_type = api_type_from_name(name)

    # Try to get the type name.
    if api_type in {"class", "instance"}:
        return get_class_or_instance_type(name)

    # Try to get something useful other than "statement".
    elif api_type == "statement":
        return get_statement_type(name) or api_type

    return api_type


class VirtualInstance(Aggregation, CompiledInstance):
    is_stub         = falsy_noargs
    is_instance     = truthy_noargs
    inference_state = state

    parent_context  = _forwarder("class_value.parent_context")

    def _as_context(self):
        return CompiledContext(self)

    class_value: "VirtualValue"       = _named_index(0)
    _arguments:   AbstractArguments   = _named_index(1)

    def get_filters(self, origin_scope=None, include_self_names=True):
        yield from self.class_value.get_filters(origin_scope=origin_scope, is_instance=True)

    def py__call__(self, arguments):
        return self.class_value.parent_value.virtual_call(arguments, self)

    def py__simple_getitem__(self, index):
        return self.class_value.virtual_getitem(index, arguments=self._arguments)

    py__getitem__ = py__simple_getitem__

    def py__iter__(self, *args, **unused):
        if instance := self.py__simple_getitem__(0):
            return Values((LazyKnownValues(instance),))
        return NO_VALUES

    def __repr__(self):
        return f"{type(self).__name__}({self.class_value._get_object_name()})"

    def py__doc__(self):
        return self.class_value.py__doc__()

    # This is how ``bpy.props.CollectionProperty(type=cls)`` is inferred.
    def execute_annotation(self):
        return self.class_value.virtual_call(arguments=self._arguments, instance=self)


class VirtualMixin:
    def get_filters(self: CompiledValue, is_instance=False, origin_scope=None):
        yield VirtualFilter((self, is_instance))

    def get_filter_get(self, name_str: str, is_instance):
        if name_str in self.members:
            return (VirtualName((self, name_str, is_instance)),)
        return ()

    def get_filter_values(self, is_instance=False):
        return list(map(VirtualName, zip(repeat(self), self.members)))

    @lazy_overwrite
    def members(self):
        return self.get_members()

    def get_members(self):
        return dict.fromkeys(dir(self.obj))

    def infer_name(self, name: "VirtualName"):
        from .tools import make_compiled_value
        value = make_compiled_value(getattr(self.obj, name.string_name), self.as_context())
        return Values((value,))

    def _get_object_name(self):
        try:
            return self.obj.__name__
        except:
            return type(self.obj).__name__


class VirtualModule(VirtualMixin, CompiledModule):
    string_names = ()
    inference_state = state
    parent_context  = None
    api_type = "module"

    def __init__(self, module_obj):
        self.obj = module_obj
        self.access_handle = get_handle(module_obj)

    def get_submodule_names(self, only_modules=False):
        return get_submodule_names(self, only_modules=only_modules)

    def __repr__(self):
        return f"<{type(self).__name__} {self.obj}>"

    def py__name__(self):
        return ".".join(self.string_names)

    def py__package__(self):
        return list(self.string_names)

    def get_signatures(self):
        return []

    def get_members(self):
        return dict(self.obj.__dict__)


virtual_overrides = {}


class VirtualName(Aggregation, CompiledName):
    _inference_state = state

    parent_value: "VirtualValue" = _named_index(0)
    string_name:   str           = _named_index(1)
    is_instance:   bool          = _named_index(2)

    def _get_qualified_names(self):
        return None

    @property
    def parent_context(self):
        return self.parent_value.as_context()

    def py__doc__(self):
        for value in self.infer():
            return value.py__doc__()
        return ""

    def docstring(self, raw=False, fast=True):
        return "VirtualName docstring goes here"

    @state_cache
    def infer(self):
        return self.parent_value.infer_name(self)

    @property
    def api_type(self):
        for value in self.infer():
            return value.api_type
        return "instance"

    def __repr__(self):
        cls_name = get_type_name(self.__class__)
        return f"{cls_name}({self.parent_value._get_object_name()}.{self.string_name})"


class VirtualValue(Aggregation, VirtualMixin, CompiledValue):
    is_compiled  = truthy_noargs
    is_class     = truthy_noargs

    is_namespace = falsy_noargs
    is_instance  = falsy_noargs
    is_module    = falsy_noargs
    is_stub      = falsy_noargs

    is_builtins_module = falsy_noargs

    inference_state = state
    instance_cls = VirtualInstance

    obj                          = _named_index(0)
    parent_value: "VirtualValue" = _named_index(1)

    def virtual_call(self, arguments=NoArguments, instance=None):
        if instance:
            return NO_VALUES
        return Values((self.instance_cls((self, arguments)),))

    # Should be overridden by subclasses.
    def virtual_getitem(self, index, arguments=NoArguments):
        return NO_VALUES

    @inline
    def __init__(self, elements):
        """elements: A tuple of object and parent value."""
        return Aggregation.__init__

    @inline
    def py__call__(self, arguments):
        return _forwarder("virtual_call")

    @property
    def access_handle(self):
        return get_handle(self.obj)

    @property
    def parent_context(self):
        return self.parent_value.as_context()

    def get_signatures(self):
        return ()

    def py__doc__(self):
        return "VirtualValue py__doc__ (override me)"

    def py__name__(self):
        try:
            return self.obj.__name__
        except:
            return None

    def get_qualified_names(self):
        return ()

    # XXX: Override me.
    def get_param_names(self):
        return []

    # Subclasses override this.
    def get_members(self):
        return get_mro_dict(self.obj)

    def __repr__(self):
        return f"{_repr(self)}({self._get_object_name()})"


class VirtualFilter(Aggregation, CompiledValueFilter):
    _inference_state = state

    compiled_value: CompiledValue = _named_index(0)
    is_instance:     bool         = _named_index(1)

    def get(self, name_str: str):
        return self.compiled_value.get_filter_get(name_str, self.is_instance)

    def values(self):
        return self.compiled_value.get_filter_values(self.is_instance)

    def __repr__(self):
        return f"{_repr(self)}({_repr(self.compiled_value)})"


class VirtualFunction(VirtualValue):
    api_type = "function"

    def py__doc__(self):
        try:
            return self.obj.__doc__
        except:
            return ""


# Implements FileIO for bpy.types.Text so jedi can use diff-parsing on them.
# This is used by several optimization modules.
class BpyTextBlockIO(Variadic, KnownContentFileIO, FileIOFolderMixin):
    get_last_modified = None.__init__  # Dummy

    read = _unbound_getter("_content")
    path = _variadic_index(0)


class BpyTextModule(Variadic, ModuleValue):
    string_names = ("__main__",)

    inference_state = state
    parent_context  = None
    _is_package     = False

    is_stub     = falsy_noargs
    is_package  = falsy_noargs
    is_module   = truthy_noargs

    _path: Path           = _forwarder("file_io.path")
    code_lines: list[str] = _forwarder("file_io._content")
    file_io: BpyTextBlockIO
    context: "BpyTextModuleContext"

    # Overrides py__file__ to return the relative path instead of the
    # absolute one. On unix this doesn't matter. On Windows it erroneously
    # prefixes a drive to the beginning of the path:
    # '/MyText' -> 'C:/MyText'.
    py__file__ = _unbound_getter("_path")
    as_context = _unbound_getter("context")

    file_io    = _variadic_index(0)

    @lazy_overwrite
    def context(self):
        return BpyTextModuleContext(self)


class BpyTextModuleContext(Variadic, MixedModuleContext):
    inference_state = state
    mixed_values = ()
    filters = None

    is_class  = falsy_noargs
    is_stub   = falsy_noargs
    is_module = truthy_noargs
    is_builtins_module = falsy_noargs
    predefined_names = types.MappingProxyType({})

    _value: BpyTextModule = _variadic_index(0)

    def get_filters(self, until_position=None, origin_scope=None):
        if not self.filters:
            # Skip the merged filter.
            # It's pointless since we're already returning A LIST OF FILTERS.
            tree_filter   = MixedParserTreeFilter(self, None, until_position, origin_scope)
            global_filter = GlobalNameFilter(self)
            self.filters  = [tree_filter, global_filter, *self._value.iter_star_filters()]
        yield from self.filters

    def py__getattribute__(self, name_or_str, name_context=None, position=None, analysis_errors=True):
        if namedef := find_definition(name_or_str):
            return tree_name_to_values(state, self, namedef)

        return super().py__getattribute__(name_or_str, name_context, position, analysis_errors)


class TextSession(Variadic):
    text_id: int            = _variadic_index(0)
    module:  BpyTextModule  = _variadic_index(1)
    file_io: BpyTextBlockIO = _forwarder("module.file_io")

    @lazy_overwrite
    def code(self) -> str | None:
        return None

    def update_from_text(self, text):
        last_code = self.code
        self.code = text.as_string()
        if self.code != last_code:
            self.file_io._content = ensure_blank_eol(self.code.splitlines(True))
            self.module.tree_node = state.grammar.parse(self.code, file_io=self.file_io, cache_path=self.file_io.path)

        # Clear filters to reset ``until_position`` since last time.
        self.module.context.filters = None


@inline
class interpreter(Script):
    _inference_state = state

    __init__ = object.__init__
    __repr__ = object.__repr__

    session: TextSession
    context      = _forwarder("session.module.context")
    _code_lines  = _forwarder("_file_io._content")
    _file_io     = _forwarder("session.file_io")
    _module_node = _forwarder("session.module.tree_node")

    # Needed by self.get_signatures.
    _get_module_context = _unbound_getter("context")

    def set_session(self, text: bpy.types.Text):
        self.session = sessions[text.id]
        self.session.update_from_text(text)

    def complete(self, text: bpy.types.Text):
        if not runtime.is_reset:
            reset_state()

        add_pending_state_reset()
        self.set_session(text)

        line, column = text.cursor_focus
        return Completion(
            state, self.context, self._code_lines, (line + 1, column),
            self.get_signatures, fuzzy=settings.use_fuzzy_search).complete()


def complete(text):
    from .opgroup.interpreter import _use_new_interpreter

    runtime.is_reset = False
    if _use_new_interpreter:
        return interpreter.complete(text)
    
    from jedi.api import Interpreter
    line, column = get_cursor_focus(text)
    return Interpreter(text.as_string(), []).complete(line + 1, column)


@inline
class cached_builtins:
    __slots__ = ("__dict__",)

    def __getattr__(self, name: str, _b=set(__builtins__)):
        if name in _b:
            return self.__dict__.setdefault(name, builtin_from_name(state, name))
        raise AttributeError(name)


@inline
def get_builtin_value(name_str: str) -> ClassValue:
    return partial(getattr, cached_builtins)


class AggregateTreeNameDefinition(Aggregation, TreeNameDefinition):
    __slots__ = ()

    parent_context = _named_index(0)
    tree_name      = _named_index(1)



class AggregateTreeArguments(Aggregation, TreeArguments):
    __slots__ = ()

    _inference_state = state

    context       = _named_index(0)
    argument_node = _named_index(1)

    @lazy_overwrite
    def trailer(self):
        return self[2]


class AggregateLazyTreeValue(Aggregation, LazyTreeValue):
    __slots__ = ()

    context = _named_index(0)
    data    = _named_index(1)

    min = 1
    max = 1

    # Needed because of LazyTreeValue.infer.
    _predefined_names = types.MappingProxyType({})


class VariadicLazyKnownValues(Variadic, LazyKnownValues):
    __slots__ = ()

    min = 1
    max = 1

    data = _variadic_index(0)


@inline
def call_py_getattr(*args, **kw):
    return partial(methodcaller, "py__getattribute__")


class Values(frozenset, ValueSet):
    __slots__ = ()

    # We need to override ValueSet's __bool__. It's essentially shit.
    # And we can't just "delete" an inherited method.
    _length = property(methodcaller("__len__"))
    __bool__ = _forwarder("_length.__bool__")
    __init__  = frozenset.__init__
    __repr__  = ValueSet.__repr__
    _from_frozen_set = object.__new__

    # ``_set`` refers to the aggregate instance itself.
    # This makes it compatible with ValueSet methods.
    _set = _forwarder("__init__.__self__")

    @classmethod
    def from_sets(cls, sets):
        return cls(starchain(sets))

    def __or__(self, other):
        return Values(starchain((self, other)))

    def py__getattribute__(self: "Values", *args, **kw):
        return Values(starchain(map(call_py_getattr(*args, **kw), self)))


class AggregateExecutedParamName(Aggregation, ExecutedParamName):
    __slots__ = ()

    function_value = _named_index(0)
    arguments      = _named_index(1)
    param_node     = _named_index(2)
    _lazy_value    = _named_index(3)
    _is_default    = _named_index(4)

    tree_name = _forwarder("param_node.name")

    @lazy_overwrite
    def parent_context(self):
        return self.function_value.get_default_param_context()


class AggregateCompiledValueFilter(Aggregation, CompiledValueFilter):
    __slots__ = ()

    _inference_state = state
    compiled_value   = _named_index(0)
    is_instance      = _named_index(1)


class AggregateStubName(Aggregation, StubName):
    parent_context = _named_index(0)
    tree_name      = _named_index(1)

    def __repr__(self):
        if self.start_pos is None:
            return '<%s: string_name=%s>' % ("StubName", self.string_name)
        return '<%s: string_name=%s start_pos=%s>' % ("StubName",
                                                      self.string_name, self.start_pos)


class AggregateClassName(Aggregation, ClassName):
    _class_value      = _named_index(0)
    tree_name         = _named_index(1)
    parent_context    = _named_index(2)
    _apply_decorators = _named_index(3)

    def __repr__(self):
        if self.start_pos is None:
            return '<%s: string_name=%s>' % ("ClassName", self.string_name)
        return '<%s: string_name=%s start_pos=%s>' % ("ClassName",
                                                      self.string_name, self.start_pos)



class AggregateSelfName(Aggregation, SelfName):
    _instance     = _named_index(0)
    class_context = _named_index(1)
    tree_name     = _named_index(2)



# CompiledName, but better.
# - Tuple-initialized to skip the construction overhead
# - Getattr static happens only when we actually need it.
class AggregateCompiledName(Aggregation, CompiledName):
    _inference_state = state

    _parent_value  = _named_index(0)
    string_name    = _named_index(1)

    @property
    def parent_context(self):
        return self._parent_value.as_context()

    def py__doc__(self):
        for value in self.infer():
            return value.py__doc__()
        return ""

    @property
    def api_type(self):
        for value in self.infer():
            return value.api_type
        return "instance"

    @state_cache
    @inline
    def infer_compiled_value(self):
        return CompiledName.infer_compiled_value.__closure__[0].cell_contents


class AggregateTreeSignatureBase(Aggregation, TreeSignature):
    def bind(self, value):
        return AggregateBoundTreeSignature((value, self._function_value, True))

    matches_signature = state_cache(TreeSignature.matches_signature)
    get_param_names   = state_cache_kw(TreeSignature.get_param_names)


class AggregateTreeSignature(AggregateTreeSignatureBase):
    value           = _named_index(0)
    _function_value = _forwarder("value")
    is_bound        = False


class AggregateBoundTreeSignature(AggregateTreeSignatureBase):
    value           = _named_index(0)
    _function_value = _named_index(1)
    is_bound        = True


class AggregateBoundMethod(Aggregation, BoundMethod):
    instance       = _named_index(0)
    _class_context = _named_index(1)
    _wrapped_value = _named_index(2)

    def get_signature_functions(self):
        args = zip(repeat(self.instance),
                   repeat(self._class_context),
                   self._wrapped_value.get_signature_functions())
        return map(AggregateBoundMethod, args)
