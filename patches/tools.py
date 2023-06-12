# This module implements various utilities for Suggestions.

from jedi.inference.compiled.value import CompiledValue, CompiledValueName, create_cached_compiled_value
from jedi.inference.value.instance import CompiledInstance, ValueSet
import jedi.api as api

from parso.python.tree import Name
from parso.grammar import load_grammar
from parso.tree import BaseNode

from textension.utils import factory, namespace, inline, _check_type
import textension

import os
import collections
from typing import Any


# Jedi's own module cache is broken. It stores empty value sets.
class StateModuleCache(dict):
    __getitem__ = None  # Don't allow subscript. Jedi doesn't use it.

    def add(self, string_names: tuple[str], value_set):
        # These checks are for when jedi is being a dummy
        # and caches something that should not be cached.
        _check_type(string_names, tuple)
        _check_type(value_set, ValueSet)

        assert string_names, "Empty tuple"
        if not value_set:
            print("Warning: Empty ValueSet for", string_names)

        all(_check_type(s, str) for s in string_names)

        self[string_names] = value_set


# The inference state is made persistent, simplifying direct object access
# and eliminates use of inference state weakrefs and other ridiculous
# indirections that add to the overhead. We only use Interpreter anyway.

# Jedi allows getattr on descriptors if we use the Interpreter class. This
# is unsafe, even if disallowing it complicates things.
# api.Interpreter._allow_descriptor_getattr_default = False

project     = api.Project(os.path.dirname(textension.__file__))
environment = api.InterpreterEnvironment()
state       = api.InferenceState(project, environment, None)

state.grammar = state.latest_grammar = load_grammar()
state.module_cache = StateModuleCache()

api.InferenceState.__new__  = lambda *args, **kw: state
api.InferenceState.__init__ = object.__init__

state.analysis = collections.deque(maxlen=25)
state.allow_descriptor_getattr = True
# Use external memoize cache so cached functions that rely on it can
# access it even before the inference state exists.
state.memoize_cache = collections.defaultdict(dict)


runtime = namespace(context_rna=None)


# Compiled value overrides used for aiding type inference.
_descriptor_overrides = collections.defaultdict(dict)
_rtype_overrides = {}
_value_overrides = {}


@inline
def is_basenode(node) -> bool:
    return BaseNode.__instancecheck__


@inline
def is_namenode(node) -> bool:
    return Name.__instancecheck__


@inline
def is_funcdef(node) -> bool:
    from parso.python.tree import Function
    return Function.__instancecheck__


@inline
def is_classdef(node) -> bool:
    from parso.python.tree import Class
    return Class.__instancecheck__


@factory
def get_handle(obj: Any):
    return state.compiled_subprocess.get_or_create_access_handle


@factory
def _filter_modules():
    from importlib.machinery import all_suffixes
    from itertools import compress
    from builtins import map, len

    suffixes = (tuple(all_suffixes()),)
    endswith = str.endswith
    index = str.index

    def _filter_modules(names: list[str]):
        for name in compress(names, map(endswith, names, suffixes * len(names))):
            yield name[:index(name, ".")]

    return _filter_modules


# Get the first method using the method resolution order.
def _get_unbound_super_method(cls, name: str):
    for base in cls.__mro__[1:]:
        try:
            return getattr(base, name)
        except AttributeError:
            continue
    raise AttributeError(f"Method '{name}' not on any superclass")


def override_method(cls: type, *, alias=None):
    def override(func, cls=cls, name=alias):
        if name is not None:
            assert isinstance(name, str)
        elif isinstance(func, property):
            name = func.fget.__name__
        else:
            name = func.__name__

        assert hasattr(cls, name), f"No method on {cls} named ``{name}``."
        setattr(cls, name, func)
        return func
    return override


from jedi.inference.compiled.access import DirectObjectAccess, create_access_path, getattr_static
from types import GetSetDescriptorType
from jedi.inference.compiled.value import CompiledValue

from itertools import compress, repeat
from operator import not_, contains


class AccessOverride(DirectObjectAccess):
    _docstring: str = ""
    _rtype = None

    def __init__(self, access: DirectObjectAccess):
        self._inference_state = access._inference_state
        self._obj = access._obj

    def py__doc__(self):
        return self._docstring or super().py__doc__()

    def get_return_annotation(self):
        if ret := super().get_return_annotation():
            return ret
        elif self._rtype:
            return create_access_path(state, self._rtype)
        return None

    def is_allowed_getattr(self, name, safe=True):
        obj = self._obj

        # Is a class.
        if not isinstance(obj, type):
            obj = type(obj)

        obj_dict = type.__dict__["__dict__"].__get__(obj)
        if isinstance(obj_dict.get(name), GetSetDescriptorType):
            return False, True  # Not an attribute, is a descriptor.
        try:
            value = object.__getattribute__(self._obj, name)
        except:
            pass
        else:
            if isinstance(value, GetSetDescriptorType):
                return False, True
        return super().is_allowed_getattr(name, safe=safe)


def set_rtype(func, rtype):
    lines = func.__doc__.splitlines()
    selector = map(not_, map(contains, lines, repeat(":rtype:")))
    return "\n".join(compress(lines, selector)) + f"\n:rtype: {rtype}"


def override_prologue(override_func):
    def wrapper(obj, value: CompiledValue):
        access = value.access_handle.access
        if not isinstance(access, AccessOverride):
            value.access_handle.access = AccessOverride(access)
        return override_func(obj, value)
    return wrapper


@override_prologue
def _rtype_override(obj, value: CompiledValue):
    value.access_handle.access._rtype = _rtype_overrides[obj]


@override_prologue
def _descriptor_override(obj, value: CompiledValue):
    pass
    # Does nothing for now. The object just needs to exist in _descriptor_overrides.


def _add_rtype_overrides(data):
    for obj, rtype in data:
        assert not (obj in _value_overrides or obj in _rtype_overrides)
        _value_overrides[obj] = _rtype_override
        _rtype_overrides[obj] = rtype


def _add_descriptor_overrides(data):
    for descriptor, rtype in data:
        obj = descriptor.__objclass__
        # assert obj not in _value_overrides, (obj, _value_overrides)
        # assert not (obj in _value_overrides or descriptor.__name__ in _descriptor_overrides.get(obj, ()))
        _value_overrides[obj] = _descriptor_override
        _descriptor_overrides[obj][descriptor.__name__] = rtype


def make_compiled_value(obj, context):
    handle = get_handle(obj)
    return create_cached_compiled_value(state, handle, context)


def make_compiled_name(obj, context, name):
    value = make_compiled_value(obj, context)
    return CompiledValueName(value, name)


def make_instance(obj, context, arguments=None):
    value = make_compiled_value(obj, context)
    return CompiledInstance(state, context, value, arguments)


def make_instance_name(obj, context, name):
    instance = make_instance(obj, context)
    return CompiledValueName(instance, name)


def add_value_override(obj, hook):
    assert obj not in _value_overrides
    _value_overrides[obj] = hook


def ensure_blank_eol(lines: list[str]):
    try:
        if lines[-1][-1] is "\n":
            return lines + [""]
        return lines
    except:  # Assume IndexError
        return lines + [""]
