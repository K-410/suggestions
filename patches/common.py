# This module implements classes extending Jedi's inference capability.

from jedi.inference.compiled.value import (CompiledValue, CompiledValueFilter,
    CompiledValueName, CompiledModule)
from jedi.inference.value.instance import CompiledInstance
from jedi.inference.context import CompiledContext
from jedi.cache import memoize_method
from jedi.inference.base_value import NO_VALUES, ValueSet
from jedi.inference.lazy_value import LazyKnownValues

from textension.utils import PyInstanceMethod_New, _forwarder, _context, falsy_noargs, truthy_noargs
from .tools import get_handle, make_instance, state, factory, _filter_modules
from pathlib import Path
from types import ModuleType
from operator import attrgetter

from parso.file_io import KnownContentFileIO
from jedi.file_io import FileIOFolderMixin

import bpy

# For VirtualModule overrides. Assumes patch_Importer_follow was called.
Importer_redirects = {}
CompiledModule_redirects = {}


class VirtualInstance(CompiledInstance):
    is_stub      = falsy_noargs
    is_instance  = falsy_noargs

    inference_state = state
    parent_context  = _forwarder("class_value.parent_context")
    getitem_type    = _forwarder("class_value.getitem_type")

    def __init__(self, value: "VirtualValue", arguments=None):
        self._arguments  = arguments
        self.class_value = value

    def get_filters(self, origin_scope=None, include_self_names=True):
        yield from self.class_value.get_filters(origin_scope=origin_scope, is_instance=True)

    def py__call__(self, arguments):
        print("VirtualInstance.py__call__ (no values) for", self.class_value)
        return NO_VALUES

    def py__simple_getitem__(self, index):
        # XXX: This seems to happen for CollectionProperties.
        if not hasattr(self, "getitem_type"):
            print("missing getitem_type for", self)
        elif self.getitem_type is not None:
            return ValueSet((make_instance(self.getitem_type, self.parent_context),))
        return NO_VALUES

    def py__iter__(self, contextualized_node=None):
        if instance := self.py__simple_getitem__(None):
            return ValueSet((LazyKnownValues(instance),))
        return NO_VALUES

    def as_name(self, name: str):
        return VirtualName(self, name)

    @property
    def name(self):
        return VirtualName(self, self.class_value.name.string_name)


class VirtualFilter(CompiledValueFilter):
    values = _forwarder("mapping.values")

    def map_values(self, value, is_instance):
        return {v.string_name: v for v in super().values()}

    def get(self, name: str):
        try:
            return (self.mapping[name],)
        except KeyError:
            return ()

    @property
    @memoize_method
    def mapping(self):
        return self.map_values(self.compiled_value, self.is_instance)

    def _create_name(self, name):
        return VirtualName(self.compiled_value, name)


class VirtualValue(CompiledValue):
    is_compiled  = truthy_noargs
    is_class     = truthy_noargs

    is_namespace = falsy_noargs
    is_instance  = falsy_noargs
    is_module    = falsy_noargs
    is_stub      = falsy_noargs

    is_builtins_module = falsy_noargs
    as_context   = PyInstanceMethod_New(CompiledContext)
    obj          = _forwarder("access_handle.access._obj")

    inference_state = state
    _api_type    = "unknown"

    filter_cls   = VirtualFilter
    instance_cls = VirtualInstance

    def __init__(self, obj, context):
        self.value_cls = self.__class__
        self.access_handle  = get_handle(obj)
        self.parent_context = context

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

    def py__call__(self, arguments):
        return ValueSet((self.instance_cls(self, arguments),))

    def get_filters(self, is_instance=False, origin_scope=None):
        return (self.filter_cls(state, self, is_instance),)

    # XXX: Override me.
    def get_param_names(self):
        return []

    def _as_context(self):
        return CompiledContext(self)


class VirtualModule(CompiledModule):
    def __init__(self, module_obj):
        assert isinstance(module_obj, ModuleType)
        self.inference_state = state
        self.parent_context = None
        self.obj = module_obj
        self.access_handle = get_handle(module_obj)

    def get_submodule_names(self, only_modules=False):
        return get_submodule_names(self, only_modules=only_modules)

    def py__getattribute__(self, name_or_str, name_context=None, position=None, analysis_errors=True):
        return super().py__getattribute__(name_or_str, name_context, position, analysis_errors)


class VirtualName(CompiledValueName):
    _inference_state = state
    parent_context   = _forwarder("_value.parent_context")

    value_type = VirtualValue

    def __init__(self, value: VirtualValue, name: str):
        self._value = value
        self.string_name = name

    def py__doc__(self):
        return self._value.py__doc__()

    def docstring(self, raw=False, fast=True):
        return "VirtualName docstring goes here"

    @memoize_method
    def infer_compiled_value(self):
        return self.infer()

    def infer(self):
        return ValueSet((self._value,))
    
    @classmethod
    def from_object(cls, obj, context, name, instance=False):
        value = cls.value_type(obj, context)
        if instance:
            value = value.instance
        return cls(value, name)

    @property
    def api_type(self):
        return self._value.api_type


# Implements FileIO for bpy.types.Text so jedi can use diff-parsing on them.
# This is used by several optimization modules.
class BpyTextBlockIO(KnownContentFileIO, FileIOFolderMixin):
    get_last_modified = None.__init__  # Dummy

    read = PyInstanceMethod_New(attrgetter("_content"))

    def __init__(self, content: list[str]):
        try:
            # We can't pass the text to the initializer because we want to
            # make a compatible interface.
            self.path = Path("/".join((bpy.data.filepath, _context.edit_text.name)))
        except AttributeError:
            self.path = None
        
        self._content = content


@factory
def get_mro_dict() -> dict:
    from builtins import type, map, reversed, isinstance
    from functools import reduce
    from operator import or_

    get_dict = type.__dict__["__dict__"].__get__
    get_mro = type.__dict__["__mro__"].__get__

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
    from itertools import repeat

    module_dict = ModuleType.__dict__["__dict__"].__get__
    module_dir  = ModuleType.__dir__

    def get_submodule_names(self, only_modules=True):
        result = []
        name = self.py__name__()

        if name not in {"builtins", "typing"}:
            m = self.obj
            assert isinstance(m, ModuleType)

            exports = set()
            # It's possible we're passing non-module objects.
            # if isinstance(m, ModuleType):
            md = module_dict(m)
            exports.update(module_dir(m) + list(md))

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



# This version differs from the stock ``_check_flows`` by:
# - Doesn't break after first hit. The stock version is designed to process
#   like-named sequences. Here we process *all* names in one go.
# - No pre-sorting. It's not applicable unless we break out of the loop.
@factory
def _check_flows(self, names):
    from jedi.inference.flow_analysis import reachability_check, UNREACHABLE

    def _check_flows(self, names):
        context = self._node_context
        value_scope = self._parser_scope
        origin_scope = self._origin_scope
        for name in names:
            if reachability_check(context=context,
                                  value_scope=value_scope,
                                  node=name,
                                  origin_scope=origin_scope) is not UNREACHABLE:
                yield name
    return _check_flows
