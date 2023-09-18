"""This module implements various utilities for Suggestions."""

from jedi.inference.compiled.subprocess import functions
from jedi.inference.compiled.value import create_cached_compiled_value
from jedi.inference.value.instance import CompiledInstance, ValueSet
import jedi.api as api
from jedi.inference.compiled.access import DirectObjectAccess, create_access_path
from types import GetSetDescriptorType

from parso.grammar import load_grammar

from textension.utils import namespace, inline, _check_type, consume
import textension

import os
import sys
import collections
from typing import Any
from types import FunctionType, MethodType


# Jedi's own module cache is broken. It stores empty value sets.
class StateModuleCache(dict):
    __getitem__ = None  # Don't allow subscript. Jedi doesn't use it.

    def add(self, string_names: tuple[str], value_set):
        # If jedi is being a dummy, I want to know it here.
        _check_type(string_names, tuple)
        _check_type(value_set, ValueSet, frozenset)

        assert string_names, "Empty tuple"

        consume(_check_type(s, str) for s in string_names)
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

# Do not allow jedi to thrash the disk.
state.do_dynamic_params_search = False

state.grammar = state.latest_grammar = load_grammar()
state.module_cache = StateModuleCache()
state.get_sys_path = project._get_sys_path = lambda *_, **__: sys.path[:]

# Remove partial indirection, we don't use subprocesses.
state.compiled_subprocess.__dict__.update(
    {k: MethodType(v, state)
     for k, v in functions.__dict__.items()
     if not k.startswith("_") and isinstance(v, FunctionType)}
)

api.InferenceState.__new__  = lambda *args, **kw: state
api.InferenceState.__init__ = object.__init__

state.analysis = collections.deque(maxlen=25)
state.allow_descriptor_getattr = True
state.memoize_cache = {}

# Used by _InferenceStateProcess
state._access_handles = {}

runtime = namespace(context_rna=None, data_rna=None)


# Compiled value overrides used for aiding type inference.
_descriptor_overrides = {}
_rtype_overrides = {}
_value_overrides = {}
_virtual_overrides = {}


@inline
def get_handle(obj: Any):
    return state.compiled_subprocess.get_or_create_access_handle







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
            return False, True, None  # Not an attribute, is a descriptor.
        try:
            value = object.__getattribute__(self._obj, name)
        except:
            pass
        else:
            if isinstance(value, GetSetDescriptorType):
                return False, True, None
        return super().is_allowed_getattr(name, safe=safe)


def override_prologue(override_func):
    def wrapper(obj, value):
        access = value.access_handle.access
        if not isinstance(access, AccessOverride):
            value.access_handle.access = AccessOverride(access)
        override_func(obj, value)
        return value
    return wrapper


@override_prologue
def _rtype_override(obj, value):
    value.access_handle.access._rtype = _rtype_overrides[obj]


@override_prologue
def _descriptor_override(obj, value):
    # Does nothing for now. The object just needs to exist in _descriptor_overrides.
    pass


def _add_rtype_overrides(data):
    for obj, rtype in data:
        assert not (obj in _value_overrides or obj in _rtype_overrides)
        _value_overrides[obj] = _rtype_override
        _rtype_overrides[obj] = rtype


def _add_descriptor_overrides(data):
    for descriptor, rtype in data:
        _descriptor_overrides[descriptor] = rtype


def make_compiled_value(obj, context):
    handle = get_handle(obj)
    return create_cached_compiled_value(state, handle, context)


def set_virtual_override(obj: object, virtual_type: type):
    """Set a single override."""
    _virtual_overrides[obj] = virtual_type


def add_virtual_overrides(overrides: tuple[tuple[object, type]]):
    """Add a sequence of tuple pairs of object and virtual value type."""
    _virtual_overrides.update(overrides)


def ensure_blank_eol(lines: list[str]):
    try:
        if lines[-1][-1] is "\n":
            return lines + [""]
        return lines
    except:  # Assume IndexError
        return lines + [""]
