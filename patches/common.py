# This module implements classes extending Jedi's inference capability.

from jedi.inference.compiled.value import (
    CompiledValue, CompiledValueFilter, CompiledValueName, CompiledModule)
from jedi.inference.lazy_value import LazyKnownValues, NO_VALUES, ValueSet
from jedi.inference.context import CompiledContext, GlobalNameFilter
from jedi.inference.value import CompiledInstance, ModuleValue
from jedi.inference.names import TreeNameDefinition
from jedi.api.interpreter import MixedModuleContext, MixedParserTreeFilter
from parso.python.tree import Name
from parso.tree import BaseNode

from parso.file_io import KnownContentFileIO
from jedi.file_io import FileIOFolderMixin

from jedi.api import Script, Completion
from itertools import repeat
from operator import attrgetter
from pathlib import Path
from typing import Union

from textension.utils import (
    _forwarder, _named_index, _unbound_getter, consume, falsy_noargs, inline,
     set_name, truthy_noargs, instanced_default_cache, _descriptor, _TupleBase)

from .tools import state, get_handle, factory, is_namenode, is_basenode, ensure_blank_eol

import bpy


# For VirtualModule overrides. Assumes patch_Importer_follow was called.
Importer_redirects = {}
CompiledModule_redirects = {}

state_values = state.memoize_cache.values()
get_start_pos = attrgetter("line", "column")
get_cursor_focus = attrgetter("select_end_line_index", "select_end_character")


# Used by various optimizations.
for cls in iter(node_types := [BaseNode]):
    node_types += cls.__subclasses__()
node_types: frozenset[BaseNode] = frozenset(node_types)


def yield_filters_once(func):
    memo = state.memoize_cache[func] = {}
    def wrapper(self, is_instance=False, origin_scope=None):
        if (key := (self, is_instance)) in memo:
            return
        memo[key] = result = list(func(self, is_instance))
        yield from result
    return wrapper


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
    sess = TextSession()
    sess.text_id = text_id

    sess.module = BpyTextModule()
    sess.module.context = BpyTextModuleContext(sess.module)
    sess.module.file_io = BpyTextBlockIO()
    sess.file_io.path = Path(f"Text({text_id})")

    self[text_id] = sess
    return sess


def reset_state():
    # The memoize cache stores a dict of dicts keyed to functions.
    # The first level is never cleared. This lowers the cache overhead.
    consume(map(dict.clear, state_values))
    state.reset_recursion_limitations()
    state.inferred_element_counts = {}


def state_cache(func):
    memo = state.memoize_cache[func] = {}
    @set_name(func.__name__ + " (state_cache)")
    def wrapper(*args):
        if args not in memo:
            memo[args] = func(*args)
        return memo[args]
    return wrapper


def state_cache_kw(func):
    from textension.utils import dict_items
    from builtins import tuple
    memo = state.memoize_cache[func] = {}
    @set_name(func.__name__ + " (state_cache_kw)")
    def wrapper(*args, **kw):
        key = (args, tuple(dict_items(kw)))
        if key not in memo:
            memo[key] = func(*args, **kw)
        return memo[key]
    return wrapper


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
    namedefs = []
    pool  = scope.children[:]


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
                elif name.type == "testlist_star_expr":
                    namedefs += name.children[::2]

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

        elif n.type == "for_stmt":

            # Add the suite to pool.
            pool += n.children[-1].children

            name = n.children[1]
            if name.type == "exprlist":
                namedefs += filter(is_namenode, name.children)
            else:
                assert name.type == "name"
                namedefs += name,
        else:
            pool += n.children

    # Get definitions for the rest.
    for n in filter(is_namenode, pool):
        if n.get_definition(include_setitem=True):
            namedefs += n,

    return namedefs


def find_definition(context, ref: Name):
    if namedef := get_definition(ref):
        return TreeNameDefinition(context, namedef)

    # Inlined position adjustment from _get_global_filters_for_name.
    # if position:
    #     n = ref
    #     lambdef = None
    #     while n := n.parent:
    #         type = n.type
    #         if type in {"classdef", "funcdef", "lambdef"}:
    #             if type == "lambdef":
    #                 lambdef = n
    #             elif position < n.children[-2].start_pos:
    #                 if not lambdef or position < lambdef.children[-2].start_pos:
    #                     position = n.start_pos
    #                 break

    node = ref
    while node := node.parent:
        # Skip the dot operator.
        if node.type != "error_node":
            for name in filter(is_namenode, node.children):
                if name.value == ref.value and name is not ref:
                    return TreeNameDefinition(context, name)
    return None


def get_definition(ref: Name):
    p = ref.parent
    type  = p.type
    value = ref.value

    if type in {"funcdef", "classdef", "except_clause"}:

        # self is the class or function name.
        children = p.children
        if value == children[1].value:  # Is the function/class name definition.
            return children[1]

        # self is the e part of ``except X as e``.
        elif type == "except_clause" and value == children[-1].value:
            return children[-1]

    while p:
        if p.type in definition_types:
            for n in p.get_defined_names(True):
                if value == n.value:
                    return n
        elif p.type == "file_input":
            if n := get_module_definition_by_name(p, value):
                return n
        p = p.parent


@state_cache
def get_module_definition_by_name(module, string_name):
    for name in get_all_module_names(module):
        if name.value == string_name and name.get_definition(include_setitem=True):
            return name


@state_cache
def get_all_module_names(module):
    pool = [module]
    for n in filter(is_basenode, pool):
        pool += [n.children[1]] if n.type in {"classdef", "funcdef"} else n.children
    return list(filter(is_namenode, pool))


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
def get_mro_dict(obj) -> dict:
    from textension.utils import get_dict, get_mro
    from functools import reduce
    from builtins import type, map, reversed, isinstance
    from operator import or_

    def get_mro_dict(obj) -> dict:
        if not isinstance(obj, type):
            obj = type(obj)
        return reduce(or_, map(get_dict, reversed(get_mro(obj))))
    return get_mro_dict


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
                names += [e]

            result = list(map(SubModuleName, repeat(self.as_context()), names))
        return result
    return get_submodule_names


# A modified version of reachability check used by various optimizations.
@factory
def trace_flow(node, origin_scope):
    from jedi.inference.flow_analysis import get_flow_branch_keyword, _break_check
    from parso.python.tree import Keyword
    from itertools import compress, count

    flows  = {"for_stmt", "if_stmt", "try_stmt", "while_stmt", "with_stmt"}
    scopes = {"classdef", "comp_for", "file_input", "funcdef",  "lambdef",
              "sync_comp_for"}

    any_scope = scopes | flows
    is_keyword = Keyword.__instancecheck__

    @state_cache
    def iter_flows(node, include_scopes):
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

    @state_cache
    def trace_flow(name, origin_scope):
        if origin_scope is not None:
            branch_matches = True
            for flow_scope in iter_flows(origin_scope, False):  # 0.5ms (274)
                if flow_scope in iter_flows(name, False):
                    node_keyword   = get_flow_branch_keyword(flow_scope, name)
                    origin_keyword = get_flow_branch_keyword(flow_scope, origin_scope)

                    if branch_matches := node_keyword == origin_keyword:
                        break

                    elif flow_scope.type == "if_stmt" or \
                        (flow_scope.type == "try_stmt" and origin_keyword == 'else' and node_keyword == 'except'):
                            return False

            if branch_matches:
                first_flow_scope = None
                for first_flow_scope in iter_flows(name, True):
                    break
                while origin_scope:
                    if first_flow_scope is origin_scope:
                        return name
                    origin_scope = origin_scope.parent

        # XXX: For testing. What's the point of break check?
        # if first_flow_scope is None:
        #     first_flow_scope = next(iter_flows(name, True), None)
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
    is_instance     = falsy_noargs
    inference_state = state

    parent_context  = _forwarder("class_value.parent_context")
    getitem_type    = _forwarder("class_value.getitem_type")

    def _as_context(self):
        return CompiledContext(self)

    def __init__(self, value: "VirtualValue", arguments=None):
        self._arguments  = arguments
        self.class_value = value

    def get_filters(self, origin_scope=None, include_self_names=True):
        yield from self.class_value.get_filters(origin_scope=origin_scope, is_instance=True)

    def py__call__(self, arguments):
        print("VirtualInstance.py__call__ (no values) for", self.class_value)
        return NO_VALUES

    # Don't try to support indexing.
    def py__simple_getitem__(self, *args, **unused):
        return self.class_value.py__simple_getitem__(0)

    py__getitem__ = py__simple_getitem__

    def py__iter__(self, *args, **unused):
        if instance := self.py__simple_getitem__(0):
            return ValueSet((LazyKnownValues(instance),))
        return NO_VALUES

    @property
    def name(self):
        value = self.class_value
        return VirtualName((value, value.name.string_name, True))

    def __repr__(self):
        return f"{type(self).__name__}({self.class_value})"

    def py__doc__(self):
        return self.class_value.py__doc__()


class VirtualModule(CompiledModule):
    inference_state = state
    parent_context  = None

    def __init__(self, module_obj):
        self.obj = module_obj
        self.access_handle = get_handle(module_obj)

    def get_submodule_names(self, only_modules=False):
        return get_submodule_names(self, only_modules=only_modules)

    def __repr__(self):
        return f"<{type(self).__name__} {self.obj}>"


virtual_overrides = {}


class VirtualName(_TupleBase, CompiledValueName):
    _value: Union["VirtualValue", "VirtualInstance"]
    _inference_state = state

    parent_value: "VirtualValue" = _named_index(0)
    string_name:   str           = _named_index(1)
    is_instance:   bool          = _named_index(2)


    @property
    def parent_context(self):
        return self.parent_value.as_context()

    def py__doc__(self):
        if value := self._value:
            return value.py__doc__()
        return ""

    def docstring(self, raw=False, fast=True):
        return "VirtualName docstring goes here"

    @state_cache
    def _infer(self):
        obj = self.parent_value.members[self.string_name]
        value_type = VirtualValue
        if obj in virtual_overrides:
            value_type = virtual_overrides[obj]

        value = value_type((obj, self.parent_value))
        if self.is_instance:
            value = value.instance
        return value

    def infer(self):
        return ValueSet((self._value,))

    infer_compiled_value = infer
    _value = _descriptor(_infer)

    @property
    def api_type(self):
        if value := self._value:
            return value.api_type
        return "unknown"

    def __repr__(self):
        return f"{self.parent_value}.{self.string_name}"


class VirtualFilter(_TupleBase, CompiledValueFilter):
    compiled_value: "VirtualValue"
    is_instance: bool

    _inference_state = state

    compiled_value = _named_index(0)
    is_instance    = _named_index(1)

    name_cls = VirtualName

    def get(self, name_str: str):
        value = self.compiled_value
        if name_str in value.members:
            return (self.name_cls((value, name_str, self.is_instance)),)
        return ()

    def values(self):
        value = self.compiled_value
        return list(map(self.name_cls, zip(repeat(value), value.members)))

    def __repr__(self):
        return f"{_repr(self)}({_repr(self.compiled_value)})"


class VirtualValue(_TupleBase, CompiledValue):
    is_compiled  = truthy_noargs
    is_class     = truthy_noargs

    is_namespace = falsy_noargs
    is_instance  = falsy_noargs
    is_module    = falsy_noargs
    is_stub      = falsy_noargs
    is_builtins_module = falsy_noargs

    inference_state = state

    _api_type    = "unknown"
    filter_cls   = VirtualFilter
    instance_cls = VirtualInstance

    parent_value: "VirtualValue"

    obj          = _named_index(0)
    parent_value = _named_index(1)

    # Just to satisfy jedi.
    @property
    def access_handle(self):
        return get_handle(self.obj)

    @property
    def parent_context(self):
        return self.parent_value.as_context()

    @property
    def api_type(self):
        return self._api_type

    @property
    def instance(self):
        return self.instance_cls(self)

    def get_signatures(self):
        return ()

    def py__doc__(self):
        return "VirtualValue py__doc__ (override me)"

    def py__name__(self):
        return "VirtualValue.py__name__ (override me)"

    def py__call__(self, arguments=None):
        return ValueSet((self.instance_cls(self, arguments),))

    @yield_filters_once
    def get_filters(self, is_instance=False, origin_scope=None):
        return (self.filter_cls((self, is_instance)),)

    def get_qualified_names(self):
        return ()

    # XXX: Override me.
    def get_param_names(self):
        return []

    def _as_context(self):
        return CompiledContext(self)

    @property
    @state_cache
    def members(self):
        return get_mro_dict(self.obj)

    def __repr__(self):
        return f"{_repr(self)}({repr(self.obj)})"


# Implements FileIO for bpy.types.Text so jedi can use diff-parsing on them.
# This is used by several optimization modules.
class BpyTextBlockIO(KnownContentFileIO, FileIOFolderMixin):
    get_last_modified = None.__init__  # Dummy
    read = _unbound_getter("_content")
    __init__ = object.__init__


class DeferredDefinition(_TupleBase, TreeNameDefinition):
    parent_context = _named_index(0)
    tree_name      = _named_index(1)


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

    __init__ = object.__init__

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
        return self.filters

    def py__getattribute__(self, name_or_str, name_context=None, position=None, analysis_errors=True):
        if namedef := find_definition(self, name_or_str):
            return namedef.infer()
        print("BpyTextModuleContext py__getattribute__ failed for", name_or_str)
        return super().py__getattribute__(name_or_str, name_context, position, analysis_errors)


class TextSession:
    text_id: int
    code:    str = None

    module:  BpyTextModule
    file_io: BpyTextBlockIO = _forwarder("module.file_io")

    def update_from_text(self, text):
        last_code = self.code
        self.code = text.as_string()
        if self.code != last_code:
            self.file_io._content = ensure_blank_eol(self.code.splitlines(True))
            self.module.tree_node = state.grammar.parse(self.code, file_io=self.file_io, cache_path=self.file_io.path)

        # Clear filters to reset since ``until_position`` since last time.
        self.module.context.filters = None


@inline
class interpreter(Script):
    _inference_state = state

    __init__ = object.__init__
    __repr__ = object.__repr__

    context      = _forwarder("session.module.context")
    _code_lines  = _forwarder("_file_io._content")
    _file_io     = _forwarder("session.file_io")
    _module_node = _forwarder("session.module.tree_node")

    # Needed by self.get_signatures.
    _get_module_context = _unbound_getter("context")

    def complete(self, text: bpy.types.Text):
        reset_state()

        self.session = sessions[text.id]
        self.session.update_from_text(text)

        line, column = get_cursor_focus(text)
        return Completion(
            state, self.context, self._code_lines, (line + 1, column),
            self.get_signatures, fuzzy=False).complete()


def complete(text):
    from .opgroup.interpreter import _use_new_interpreter

    if _use_new_interpreter:
        return interpreter.complete(text)
    
    from jedi.api import Interpreter
    line, column = get_cursor_focus(text)
    return Interpreter(text.as_string(), []).complete(line + 1, column)
