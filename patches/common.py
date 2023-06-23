# This module implements classes extending Jedi's inference capability.

from jedi.inference.compiled.value import (
    CompiledValue, CompiledValueFilter, CompiledValueName, CompiledModule)
from jedi.inference.value.instance import CompiledInstance
from jedi.inference.base_value import NO_VALUES, ValueSet
from jedi.inference.lazy_value import LazyKnownValues
from jedi.inference.context import CompiledContext
from jedi.inference.names import TreeNameDefinition, StubName
from parso.python.tree import Name

from jedi.file_io import FileIOFolderMixin
from parso.file_io import KnownContentFileIO

from itertools import repeat
from operator import attrgetter
from pathlib import Path
from types import ModuleType

from textension.utils import _unbound_getter, _forwarder, set_name, _named_index
from textension.utils import _context, falsy_noargs, truthy_noargs
from .tools import state, get_handle, make_instance, factory, is_namenode, is_basenode

import bpy


# For VirtualModule overrides. Assumes patch_Importer_follow was called.
Importer_redirects = {}
CompiledModule_redirects = {}


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

    def py__simple_getitem__(self, index):
        return self.class_value.py__simple_getitem__(index)

    def py__iter__(self, contextualized_node=None):
        if instance := self.py__simple_getitem__(None):
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

    def py__getattribute__(self, name_or_str, name_context=None, position=None, analysis_errors=True):
        return super().py__getattribute__(name_or_str, name_context, position, analysis_errors)

    def __repr__(self):
        return f"<{type(self).__name__} {self.obj}>"


# pydevd substitutes tuple subclass instances' own __repr__ with a useless
# string. This is a workaround specifically for debugging purposes.
class _pydevd_repr_override_meta(type):
    @property
    def __name__(cls):
        import sys
        for v in filter(cls.__instancecheck__, sys._getframe(1).f_locals.values()):
            return cls.__repr__(v)
        return super().__name__


class _TupleBase(tuple, metaclass=_pydevd_repr_override_meta):
    __init__ = tuple.__init__
    __new__  = tuple.__new__


def _repr(obj):
    return type.__dict__['__name__'].__get__(type(obj))


class VirtualName(_TupleBase, CompiledValueName):
    _inference_state = state

    parent_value: "VirtualValue" = _named_index(0)
    string_name:   str           = _named_index(1)
    is_instance:   bool          = _named_index(2)

    @property
    def _value(self):
        return self._infer()

    @property
    def parent_context(self):
        return self.parent_value.as_context()

    def py__doc__(self):
        if value := self.infer():
            value, = value
            return value.py__doc__()
        return ""

    def docstring(self, raw=False, fast=True):
        return "VirtualName docstring goes here"

    def infer_compiled_value(self):
        return self.infer()
    
    @state_cache
    def _infer(self):
        obj = self.parent_value.members[self.string_name]
        value = VirtualValue((obj, self.parent_value))
        if self.is_instance:
            value = value.instance
        return value

    def infer(self):
        return ValueSet((self._infer(),))

    @property
    def api_type(self):
        if value := self.infer():
            value, = value
            return value.api_type
        return "unknown"

    def __repr__(self):
        return f"{self.parent_value}.{self.string_name}"


class VirtualFilter(_TupleBase, CompiledValueFilter):
    _inference_state = state

    compiled_value: "VirtualValue" = _named_index(0)
    is_instance:     bool          = _named_index(1)

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

    obj                          = _named_index(0)
    parent_value: "VirtualValue" = _named_index(1)

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

    def get_filters(self, is_instance=False, origin_scope=None):
        yield self.filter_cls((self, is_instance))

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

    def __init__(self, content: list[str]):
        try:
            # We can't pass the text to the initializer because we want to
            # make a compatible interface.
            self.path = Path("/".join((bpy.data.filepath, _context.edit_text.name)))
        except AttributeError:
            self.path = None
        
        self._content = content


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

    def get_submodule_names(self, only_modules=True):
        result = []
        name = self.py__name__()

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
        first_flow_scope = None

        if origin_scope is not None:
            branch_matches = True
            for flow_scope in iter_flows(origin_scope, False):
                if flow_scope in iter_flows(name, False):
                    node_keyword   = get_flow_branch_keyword(flow_scope, name)
                    origin_keyword = get_flow_branch_keyword(flow_scope, origin_scope)

                    if branch_matches := node_keyword == origin_keyword:
                        break

                    elif flow_scope.type == "if_stmt" or \
                        (flow_scope.type == "try_stmt" and origin_keyword == 'else' and node_keyword == 'except'):
                            return False

            first_flow_scope = next(iter_flows(name, True), None)

            if branch_matches:
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
    if origin_scope := self._origin_scope:
        return filter(None, map(trace_flow, names, repeat(origin_scope)))
    # If origin scope is None, there's nothing to trace.
    return names


class DeferredDefinition(_TupleBase, TreeNameDefinition):
    parent_context = _named_index(0)
    tree_name      = _named_index(1)


class DeferredStubName(_TupleBase, StubName):
    parent_context = _named_index(0)
    tree_name      = _named_index(1)


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


def find_definition(context, ref, position):
    if namedef := get_definition(ref):
        return TreeNameDefinition(context, namedef)

    # Inlined position adjustment from _get_global_filters_for_name.
    if position:
        n = ref
        lambdef = None
        while n := n.parent:
            type = n.type
            if type in {"classdef", "funcdef", "lambdef"}:
                if type == "lambdef":
                    lambdef = n
                elif position < n.children[-2].start_pos:
                    if not lambdef or position < lambdef.children[-2].start_pos:
                        position = n.start_pos
                    break

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


get_start_pos = attrgetter("line", "column")


# This version doesn't include flow checks nor read or write to cache.
def get_parent_scope_fast(node):
    if scope := node.parent:
        pt = scope.type

        either = (pt == 'tfpdef' and scope.children[0] == node) or \
                 (pt == 'param'  and scope.name == node)

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
