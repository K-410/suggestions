# This module implements various utilities

import functools
import collections

from textension.utils import factory, PyFunction_SetClosure, namespace
from types import FunctionType
from typing import Any


from jedi.inference.compiled.value import CompiledValue, CompiledValueFilter, CompiledName, CompiledContext, SignatureParamName, CompiledValueName, create_cached_compiled_value
from jedi.inference.value.instance import CompiledInstance, ValueSet, NO_VALUES, CompiledBoundMethod


import jedi.api as api
import textension
import os


# The inference state is made persistent, simplifying direct object access
# and eliminates use of inference state weakrefs and other ridiculous
# indirections that add to the overhead. We only use Interpreter anyway.

# Jedi allows getattr on descriptors if we use the Interpreter class. This
# is unsafe, even if disallowing it complicates things.
# api.Interpreter._allow_descriptor_getattr_default = False

project     = api.Project(os.path.dirname(textension.__file__))
environment = api.InterpreterEnvironment()
state       = api.InferenceState(project, environment, None)

from parso.grammar import load_grammar
state.grammar = state.latest_grammar = load_grammar()

api.InferenceState.__new__ = lambda *args, **kw: state
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


# Patch a function with new closures and code object. Returns copy of the old.
def _patch_function(fn: FunctionType, new_fn: FunctionType):
    orig = _copy_function(fn)

    # Apply the closure cells from the new function.
    PyFunction_SetClosure(fn, new_fn.__closure__)

    fn.__code__ = new_fn.__code__.replace(co_name=f"{fn.__name__} ({new_fn.__name__})")
    fn.__orig__ = vars(fn).setdefault("__orig__", orig)
    return orig


# Get the first method using the method resolution order.
def _get_unbound_super_method(cls, name: str):
    for base in cls.__mro__[1:]:
        try:
            return getattr(base, name)
        except AttributeError:
            continue
    raise AttributeError(f"Method '{name}' not on any superclass")


# Make a deep copy of a function.
def _copy_function(f):
    g = FunctionType(
        f.__code__,
        f.__globals__,
        name=f.__name__,
        argdefs=f.__defaults__,
        closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


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
