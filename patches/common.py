# This module implements classes extending Jedi's inference capability.

from jedi.inference.compiled.value import (
    CompiledValue, CompiledValueFilter, CompiledName, CompiledModule)
from jedi.inference.value.instance import SelfName
from jedi.inference.syntax_tree import tree_name_to_values
from jedi.inference.value.klass import ClassValue, ClassName
from jedi.inference.lazy_value import LazyKnownValues, LazyTreeValue
from jedi.inference.arguments import AbstractArguments, TreeArguments
from jedi.inference.compiled import builtin_from_name

from jedi.inference.context import CompiledContext, GlobalNameFilter
from jedi.inference.param import ExecutedParamName
from jedi.api.interpreter import MixedModuleContext, MixedParserTreeFilter
from jedi.inference.names import TreeNameDefinition, NO_VALUES, ValueSet, StubName
from jedi.inference.value import CompiledInstance, ModuleValue
from parso.python.tree import Name
from parso.tree import BaseNode
from parso.file_io import KnownContentFileIO
from jedi.file_io import FileIOFolderMixin
from jedi.api import Script, Completion
from itertools import repeat
from operator import attrgetter, methodcaller
from pathlib import Path
import types

from functools import partial

from textension.utils import (
    _forwarder, _named_index, _unbound_getter, consume, falsy_noargs, inline,
     set_name, truthy_noargs, instanced_default_cache, Aggregation, starchain,
     lazy_overwrite, namespace, filtertrue, get_mro_dict)

from .tools import state, get_handle, factory, is_namenode, is_basenode, ensure_blank_eol
from .. import settings

import bpy


_closures = []

# For VirtualModule overrides. Assumes patch_Importer_follow was called.
Importer_redirects = {}
CompiledModule_redirects = {}

memoize_values = state.memoize_cache.values()
get_start_pos = attrgetter("line", "column")
get_cursor_focus = attrgetter("select_end_line_index", "select_end_character")

scope_types = {"classdef", "comp_for", "file_input", "funcdef",  "lambdef",
               "sync_comp_for"}

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
def get_type_name(cls: type) -> str:
    return type.__dict__["__name__"].__get__


# Used by VirtualValue.py__call__ and other inference functions where we
# don't have any arguments to unpack. For constructors.
@inline
class NoArguments(AbstractArguments):
    def unpack(self, *args, **kw):
        yield from ()


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
    module = BpyTextModule(Path(f"Text({text_id})"))
    return self.setdefault(text_id, TextSession((text_id, module)))


@inline
def _rebuild_closure_caches():
    return partial(map, partial.__call__, _closures, map(dict.__new__, repeat(dict)))


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
    consume(map(dict.clear, filtertrue(memoize_values)))

    # Persistent state means we have to clear access handles or they'll cause
    # circular references undetectable by the garbage collector and leak.
    state.compiled_subprocess._handles.clear()
    state.compiled_subprocess = state.environment.get_inference_state_subprocess(state)

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
    from operator import attrgetter
    from builtins import map

    start_pos = attrgetter("line", "column")

    def filter_until(pos: int | None, names):
        if pos:
            return compress(names, map(pos.__gt__, map(start_pos, names)))
        return names

    return filter_until


definition_types = {
    'expr_stmt',
    'sync_comp_for',
    'with_stmt',
    'for_stmt',
    'import_name',
    'import_from',
    'param',
    'del_stmt',
    'namedexpr_test',
}


@state_cache
def get_scope_name_definitions(scope):

    # Jedi allows ill-formed function/lambda constructs, so we can't assume
    # ``scope`` is even a BaseNode. This is bad, but the alternative is fixing
    # the grammar and I'm not going to touch that.
    if not is_basenode(scope):
        return ()

    namedefs = []
    pool = scope.children[:]


    # XXX: This code is for testing.
    # def check(name):
    #     if not is_namenode(name):
    #         pass
    # class L(list):
    #     def __iadd__(self, it):
    #         check(it[0])
    #         return super().__iadd__(it)
    # namedefs = L()

    for n in filter(is_basenode, pool):
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
                    namedefs += filter(is_namenode, name.children[::2])

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

        elif n.type == "decorated":
            pool += n.children[1],

        elif n.type in {"for_stmt", "if_stmt"}:
            # Add the suite to pool.
            pool += n.children[-1].children

            name = n.children[1]
            if name.type == "comparison":
                name = name.children[0]

            if name.type == "exprlist":
                namedefs += filter(is_namenode, name.children)
            elif name.type == "name":
                namedefs += name,

        elif n.type in {"error_node", "arglist"}:
            continue
        else:
            pool += n.children

    # Get definitions for the rest.
    for n in filter(is_namenode, pool):
        if n.get_definition(include_setitem=True):
            namedefs += n,

    return namedefs


def find_definition(ref: Name):
    if namedef := get_definition(ref):
        return namedef

    node = ref
    while node := node.parent:
        # Skip the dot operator.
        if node.type != "error_node":
            for name in filter(is_namenode, node.children):
                if name.value == ref.value and name is not ref:
                    return name
    return None


def get_definition(ref: Name):
    value = ref.value
    scope = ref.parent

    scope_types_ = scope_types
    while scope.type not in scope_types_:
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

    return None


@state_cache
def get_cached_scope_definitions(scope):
    from collections import defaultdict

    cache = defaultdict(list)
    scope_definitions = get_scope_name_definitions(scope)
    for namedef in scope_definitions:
        cache[namedef.value] += namedef,

    cache.default_factory = repeat(()).__next__
    return cache


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


@factory
def get_submodule_names():
    from os import listdir
    from sys import modules
    from importlib.util import find_spec
    from jedi.inference.names import SubModuleName
    from textension.utils import get_module_dict, get_module_dir
    from itertools import repeat
    from .tools import _filter_modules
    from types import ModuleType

    def get_submodule_names(self, only_modules=True):
        result = []
        name = self.py__name__()

        # Exclude these as we already know they don't have sub-modules.
        if name not in {"builtins", "typing"}:
            m = self.obj
            assert isinstance(m, ModuleType)

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
                    exports.update(_filter_modules(listdir(path)))
            except (TypeError, FileNotFoundError):
                pass

            names = []
            for e in filter(str.__instancecheck__, exports):
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


class VirtualInstance(CompiledInstance):
    is_stub         = falsy_noargs
    is_instance     = truthy_noargs
    inference_state = state

    parent_context  = _forwarder("class_value.parent_context")

    def _as_context(self):
        return CompiledContext(self)

    def __init__(self, value: "VirtualValue", arguments=None):
        self._arguments  = arguments
        self.class_value = value

    def get_filters(self, origin_scope=None, include_self_names=True):
        yield from self.class_value.get_filters(origin_scope=origin_scope, is_instance=True)

    def py__call__(self, arguments):
        return self.class_value.parent_value.virtual_call(arguments, self)

    # Don't try to support indexing.
    def py__simple_getitem__(self, *args, **unused):
        return self.class_value.py__simple_getitem__(0)

    py__getitem__ = py__simple_getitem__

    def py__iter__(self, *args, **unused):
        if instance := self.py__simple_getitem__(0):
            return Values((LazyKnownValues(instance),))
        return NO_VALUES

    @property
    def name(self):
        value = self.class_value
        return VirtualName((value, value.name.string_name, True))

    def __repr__(self):
        return f"{type(self).__name__}({self.class_value._get_object_name()})"

    def py__doc__(self):
        return self.class_value.py__doc__()


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

    @state_cache
    def infer_name_cached(self, name: "VirtualName"):
        return self.infer_name(name)

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
        return []

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

    def infer(self):
        return self.parent_value.infer_name_cached(self)

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

    def virtual_call(self, arguments, instance=None):
        if instance:
            return NO_VALUES
        return Values((self.as_instance(arguments),))

    @inline
    def __init__(self, elements):
        """elements: A tuple of object and parent value."""
        return Aggregation.__init__

    @inline
    def py__call__(self, arguments=NoArguments):
        return _forwarder("virtual_call")

    @property
    def access_handle(self):
        return get_handle(self.obj)

    @property
    def parent_context(self):
        return self.parent_value.as_context()

    @property
    def api_type(self):
        return "unknown"

    def as_instance(self, arguments=NoArguments):
        return self.instance_cls(self, arguments)

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

    def _as_context(self):
        return CompiledContext(self)

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
class BpyTextBlockIO(Aggregation, KnownContentFileIO, FileIOFolderMixin):
    get_last_modified = None.__init__  # Dummy
    read = _unbound_getter("_content")

    path = _named_index(0)


class BpyTextModule(ModuleValue):
    string_names = ("__main__",)

    inference_state = state
    parent_context  = None
    _is_package     = False

    is_stub     = falsy_noargs
    is_package  = falsy_noargs
    is_module   = truthy_noargs

    _path: Path = _forwarder("file_io.path")
    code_lines: list[str] = _forwarder("file_io._content")
    file_io: BpyTextBlockIO
    context: "BpyTextModuleContext"

    # Overrides py__file__ to return the relative path instead of the
    # absolute one. On unix this doesn't matter. On Windows it erroneously
    # prefixes a drive to the beginning of the path:
    # '/MyText' -> 'C:/MyText'.
    py__file__ = _unbound_getter("_path")

    def __init__(self, path):
        self.context = BpyTextModuleContext(self)        
        self.file_io = BpyTextBlockIO((path,))

    def as_context(self):
        return self.context


class BpyTextModuleContext(MixedModuleContext):
    inference_state = state
    mixed_values = ()
    filters = None

    is_class  = falsy_noargs
    is_stub   = falsy_noargs
    is_module = truthy_noargs
    is_builtins_module = falsy_noargs

    def __init__(self, value: BpyTextModule):
        self._value = value
        self.predefined_names = {}

    def get_filters(self, until_position=None, origin_scope=None):
        if not self.filters:
            # Skip the merged filter.
            # It's pointless since we're already returning A LIST OF FILTERS.
            tree_filter   = MixedParserTreeFilter(self, None, until_position, origin_scope)
            global_filter = GlobalNameFilter(self)
            self.filters  = [tree_filter, global_filter]
        yield from self.filters

    def py__getattribute__(self, name_or_str, name_context=None, position=None, analysis_errors=True):
        if namedef := find_definition(name_or_str):
            return tree_name_to_values(state, self, namedef)

        return super().py__getattribute__(name_or_str, name_context, position, analysis_errors)


class TextSession(Aggregation):
    text_id: int            = _named_index(0)
    module:  BpyTextModule  = _named_index(1)
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

    def complete(self, text: bpy.types.Text):
        if not runtime.is_reset:
            reset_state()

        add_pending_state_reset()

        self.session = sessions[text.id]
        self.session.update_from_text(text)

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


class AggregateLazyKnownValues(Aggregation, LazyKnownValues):
    __slots__ = ()

    min = 1
    max = 1

    data = _named_index(0)


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

    def __or__(self, x):
        if isinstance(x, ValueSet):
            x = x._set
        return frozenset.__or__(self, x)


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

    # __repr__ = _aggregate_name_repr


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
