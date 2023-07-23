# This module implements optimizations for various filters and stubs.
from jedi.inference.compiled.getattr_static import getattr_static
from jedi.inference.compiled.access import ALLOWED_DESCRIPTOR_ACCESS
from jedi.inference.compiled.value import CompiledName
from jedi.inference.gradual.stub_value import StubModuleValue, StubModuleContext, StubFilter, StubName
from jedi.inference.value.klass import ClassFilter

from textension.utils import instanced_default_cache, truthy_noargs, _named_index, Aggregation, lazy_overwrite, _forwarder
from ..common import _check_flows, state_cache, AggregateValues
from ..tools import is_basenode, is_namenode, state
from operator import attrgetter
from itertools import repeat


def apply():
    optimize_SelfAttributeFilter_values()
    optimize_ClassFilter_values()
    optimize_ClassFilter_filter()
    optimize_ClassFilter_get()
    optimize_AnonymousMethodExecutionFilter()
    optimize_ParserTreeFilter_values()
    optimize_CompiledValueFilter_values()
    optimize_AnonymousMethodExecutionFilter_values()
    optimize_ParserTreeFilter_filter()
    optimize_CompiledValue_get_filters()
    optimize_BaseTreeInstance_get_filters()
    optimize_ClassMixin_get_filters()

    optimize_StubModules()


# Pretty much same as ClassFilter, except using aggregate initialization
# and expensive values are computed on-demand.
class AggregateClassFilter(Aggregation, ClassFilter):
    _class_value   = _named_index(0)
    _context_value = _named_index(1)
    _origin_scope  = _named_index(2)
    _is_instance   = _named_index(3)

    _parser_scope  = _forwarder("_context_value.tree_node")
    until_position = None

    @lazy_overwrite
    def parent_context(self):
        return self._class_value.as_context()

    @lazy_overwrite
    def _node_context(self):
        return self._context_value.as_context()


def is_allowed_getattr(obj, name):
    try:
        attr, is_get_descriptor = getattr_static(obj, name)
    except AttributeError:
        return False, False
    else:
        if is_get_descriptor and type(attr) not in ALLOWED_DESCRIPTOR_ACCESS:
            return True, True
    return True, False


# CompiledName, but better.
# - Tuple-initialized to skip the construction overhead
# - Getattr static happens only when we actually need it.
class DeferredCompiledName(tuple, CompiledName):
    __init__ = object.__init__
    
    _inference_state = state

    _parent_value  = _named_index(0)  # parent_value
    parent_context = _named_index(1)  # parent_value.as_context()
    string_name    = _named_index(2)  # name

    def py__doc__(self):
        if value := self.infer_compiled_value():
            return value.py__doc__()
        return ""

    @property
    def api_type(self):
        if value := self.infer_compiled_value():
            return value.api_type
        return "unknown"  # XXX: Not really valid.

    def infer(self):
        has_attribute, is_descriptor = is_allowed_getattr(self._parent_value, self.string_name)
        return AggregateValues((self.infer_compiled_value(),))



# Optimizes ``values`` to not access attributes off a compiled value when
# converting them into names. Instead they are read only when inferred.
def optimize_CompiledValueFilter_values():
    from jedi.inference.compiled.value import CompiledValueFilter
    from itertools import repeat
    from ..common import state_cache

    @state_cache
    def values(self: CompiledValueFilter):
        value = self.compiled_value
        obj = value.access_handle.access._obj
        sequences = zip(repeat(value), repeat(value.as_context()), dir(obj))
        # Note: type completions is not added, because why would we?
        return list(map(DeferredCompiledName, sequences))

    CompiledValueFilter.values = values


# Optimizes SelfAttributeFilter.values to exclude stubs and limit the search
# for self-assigned definitions to within a class' methods.
def optimize_SelfAttributeFilter_values():
    from jedi.inference.value.instance import SelfAttributeFilter
    from ..tools import is_basenode, is_namenode, is_funcdef, is_classdef
    from itertools import repeat
    from parso.python.tree import Class

    def values(self: SelfAttributeFilter):
        scope = self._parser_scope

        # We only care about classes.
        if scope.__class__ is Class:
            context = self.parent_context

            # Stubs don't have self definitions.
            if not context.is_stub():
                class_nodes = scope.children[-1].children

                pool  = []
                names = []

                for n in filter(is_funcdef, class_nodes):
                    pool += n.children[-1].children

                # Recurse node trees.
                for n in filter(is_basenode, pool):
                    pool += n.children

                # Get name definitions.
                for n in filter(is_namenode, pool):
                    if n.get_definition(include_setitem=True):
                        names += n,

                names = list(self._filter(names))
                names = list(map(self.name_class, repeat(context), names))
                return names
        return []

    SelfAttributeFilter.values = values


def optimize_ClassFilter_values():
    from jedi.inference.value.klass import ClassFilter
    from parso.python.tree import Class
    from ..common import DeferredDefinition, state_cache, get_scope_name_definitions
    from builtins import list, map, zip

    stub_classdef_cache = {}

    @state_cache
    def values(self: ClassFilter):
        scope = self._parser_scope

        if scope.__class__ is Class:
            value = self._class_value
            # The class suite.
            scope = scope.children[-1]
            names = []

            if self._context_value.is_stub():
                if scope not in stub_classdef_cache:
                    stub_classdef_cache[scope] = self._filter(get_scope_name_definitions(scope))
                names = stub_classdef_cache[scope]
            else:
                names = get_scope_name_definitions(scope)

            return list(map(DeferredDefinition, zip(repeat(value), names)))
        return ()

    ClassFilter.values = values
    ClassFilter._check_flows = _check_flows


def optimize_ClassFilter_filter():
    from jedi.inference.value.klass import ClassFilter

    def _filter(self: ClassFilter, names):
        tmp = []
        for name in names:
            # TODO: Use functional style.
            if name.value.startswith("__") and not name.value.endswith("__"):
                continue
            tmp += name,
        return tmp

    ClassFilter._filter = _filter


def optimize_ClassFilter_get():
    from jedi.inference.value.klass import ClassFilter
    from ..common import get_cached_scope_definitions, DeferredDefinition

    def get(self: ClassFilter, name_str: str):
        names = get_cached_scope_definitions(self._parser_scope)[name_str]
        return list(map(DeferredDefinition, zip(repeat(self._class_value), names)))

    ClassFilter.get = get


def optimize_AnonymousMethodExecutionFilter():
    from jedi.inference.value.instance import AnonymousMethodExecutionFilter
    from ..common import get_scope_name_definitions

    def get(self: AnonymousMethodExecutionFilter, name_string):
        scope = self._parser_scope

        names = self._filter(get_scope_name_definitions(scope))
        names = [n for n in names if n.value == name_string]
        names = self._convert_names(names)
        return names

    AnonymousMethodExecutionFilter.get = get
    AnonymousMethodExecutionFilter._check_flows = _check_flows


def optimize_ParserTreeFilter_values():
    from jedi.inference.filters import ParserTreeFilter
    from ..common import get_scope_name_definitions

    def values(self: ParserTreeFilter):
        scope   = self._parser_scope

        names = get_scope_name_definitions(scope)
        names = self._filter(names)
        return self._convert_names(names)
    
    ParserTreeFilter.values = values
    ParserTreeFilter._check_flows = _check_flows


def optimize_AnonymousMethodExecutionFilter_values():
    from jedi.inference.value.instance import AnonymousMethodExecutionFilter
    from ..common import get_scope_name_definitions
    from ..tools import is_param

    def values(self: AnonymousMethodExecutionFilter):

        names = []
        scope = self._parser_scope

        # Get parameter name definitions.
        for param in filter(is_param, scope.children[2].children):
            names += param.children[0],

        # The suite.
        names += get_scope_name_definitions(scope.children[-1])
        names = self._filter(names)
        return self._convert_names(names)
    
    AnonymousMethodExecutionFilter.values = values
    AnonymousMethodExecutionFilter._check_flows = _check_flows


def optimize_ParserTreeFilter_filter():
    from jedi.inference.filters import ParserTreeFilter
    from itertools import compress
    from builtins import map
    from ..common import _check_flows

    get_start = attrgetter("start_pos")

    def _filter(self: ParserTreeFilter, names):
        if until := self._until_position:
            names = compress(names, map(until.__gt__, map(get_start, names)))

        # XXX: Needed because jedi still calls this with names from all over.
        names = [n for n in names if self._is_name_reachable(n)]

        return _check_flows(self, names)

    ParserTreeFilter._filter = _filter


def optimize_CompiledValue_get_filters():
    from jedi.inference.compiled.value import CompiledValue, CompiledValueFilter

    # XXX: Disable decorator for now. Causes issues inside class bodies.
    def get_filters(self: CompiledValue, is_instance=False, origin_scope=None):
        yield CompiledValueFilter(state, self, is_instance)

    CompiledValue.get_filters = get_filters


# Optimize to never include self filters when the tree instance is a stub.
# If it wasn't obvious, stubs do not have meaningful function/class suites.
def optimize_BaseTreeInstance_get_filters():
    from jedi.inference.value.instance import (
        _BaseTreeInstance,
        ClassFilter,
        CompiledInstanceClassFilter,
        CompiledValueFilter,
        InstanceClassFilter,
        SelfAttributeFilter,
    )

    def get_filters(self: _BaseTreeInstance, origin_scope=None, include_self_names=True):
        class_value = self.get_annotated_class_object()

        if include_self_names and not self.is_stub():
            for cls in class_value.py__mro__():
                if not cls.is_compiled():
                    yield SelfAttributeFilter(self, class_value, cls.as_context(), origin_scope)

        class_filters = class_value.get_filters(origin_scope=origin_scope, is_instance=True)

        for f in class_filters:
            if isinstance(f, ClassFilter):
                yield InstanceClassFilter(self, f)
            elif isinstance(f, CompiledValueFilter):
                yield CompiledInstanceClassFilter(self, f)
            else:
                yield f

    _BaseTreeInstance.get_filters = get_filters


def optimize_ClassMixin_get_filters():
    from jedi.inference.value.instance import InstanceClassFilter
    from jedi.inference.value.klass import ClassMixin, ClassValue
    from ..common import NoArguments, state_cache, cached_builtins


    cached_type_filter = None

    def get_cached_type_filter():
        nonlocal cached_type_filter

        if cached_type_filter is None:
            instance, = cached_builtins.type.py__call__(NoArguments)
            for f in instance.class_value.get_filters(origin_scope=None, is_instance=True):
                cached_type_filter = InstanceClassFilter(instance, f)
                break
        return cached_type_filter

    @state_cache
    def get_cached_compiled_class_filters(cls: ClassValue, is_instance: bool):
        filters = []
        filters += cls.get_filters(is_instance=is_instance)
        return filters

    def _get_filters(self, origin_scope=None, is_instance=False, include_metaclasses=True, include_type_when_class=True):

        filters = []
        if metaclasses := include_metaclasses and self.get_metaclasses():
            filters += self.get_metaclass_filters(metaclasses, is_instance)

        for cls in self.py__mro__():
            if cls.is_compiled():
                filters += get_cached_compiled_class_filters(cls, is_instance)
            else:
                filters += AggregateClassFilter((self, cls, origin_scope, is_instance)),

        if not is_instance and include_type_when_class and cached_type_filter is not self:
            filters += get_cached_type_filter(),
        return filters

    stub_class_filter_cache = {}

    def get_filters(self: ClassValue, **kw):
        if self.is_stub():
            # Stub filters are always cached.
            key = self.tree_node, tuple(kw.items())
            if key not in stub_class_filter_cache:
                stub_class_filter_cache[key] = _get_filters(self, **kw)
            return stub_class_filter_cache[key]
        return _get_filters(self, **kw)

    ClassMixin.get_filters = get_filters


class DeferredStubName(Aggregation, StubName):
    parent_context = _named_index(0)
    tree_name      = _named_index(1)


# TODO: Can this be removed in favor of cached scope name definitions?
@state_cache
def get_scope_name_strings(scope):
    namedefs = []
    pool  = scope.children[:]

    for n in filter(is_basenode, pool):
        if n.type in {"classdef", "funcdef"}:
            # Add straight to ``namedefs`` we know they are definitions.
            namedefs += n.children[1].value,

        elif n.type == "simple_stmt":
            n = n.children[0]

            if n.type == "expr_stmt":
                name = n.children[0]
                # Could be ``atom_expr``, as in dotted name. Skip those.
                if name.type == "name":
                    namedefs += name.value,
                else:
                    print("get_scope_name_strings:", repr(name.type))

            elif n.type == "import_from":
                name = n.children[3]
                if name.type == "operator":
                    name = n.children[4]

                # Only matching aliased imports are exported for stubs.
                if name.type == "import_as_names":
                    for name in name.children[::2]:
                        if name.type == "import_as_name":
                            name, alias = name.children[::2]
                            if name.value == alias.value:
                                namedefs += alias.value,

            elif n.type == "atom":
                # ``atom`` nodes are too ambiguous to extract names from.
                # Just into the pool and look for names the old way.
                pool += n.children

        elif n.type == "decorated":
            pool += n.children[1],
        else:
            pool += n.children

    # Get name definitions.
    for n in filter(is_namenode, pool):
        if n.get_definition(include_setitem=True):
            namedefs += n.value,

    keys = set(namedefs)
    exclude = set()
    for k in keys:
        if k[0] is "_" != k[:2][-1]:
            exclude.add(k)

    return keys - exclude


def get_stub_values(self: dict, stub_filter: "CachedStubFilter"):
    context = stub_filter.parent_context
    value   = context._value
    module_names = get_scope_name_strings(value.tree_node)

    names = []
    pool  = [value.tree_node]

    for n in filter(is_basenode, pool):
        pool += (n.children[1],) if n.type in {"classdef", "funcdef"} else n.children

    for n in filter(is_namenode, pool):
        if n.value in module_names:
            names += n,

    ret = stub_filter._filter(names)
    ret = self[stub_filter] = list(map(DeferredStubName, zip(repeat(context), ret)))
    return ret


stub_values_cache = instanced_default_cache(get_stub_values)


class CachedStubFilter(StubFilter):
    _until_position   = None
    _origin_scope     = None
    _parso_cache_node = None

    _check_flows = _check_flows

    def __init__(self, parent_context):
        self.parent_context = parent_context
        self._node_context  = parent_context
        self.cache = {}

    @lazy_overwrite
    def _parser_scope(self):
        return self.parent_context.tree_node

    @lazy_overwrite
    def _used_names(self):
        return self.parent_context.tree_node.get_root_node().get_used_names()

    def get(self, name):
        if name := get_module_definition_by_name(self._parser_scope, name):
            return (DeferredStubName((self.parent_context, name)),)
        return ()

    def _filter(self, names):
        return self._check_flows(names)

    def values(self: StubFilter):
        return stub_values_cache[self]


class CachedStubModuleContext(StubModuleContext):
    def get_filters(self, until_position=None, origin_scope=None):
        return list(self._value.get_filters())


definition_cache = {}


def get_module_definition_by_name(module, string_name):
    key = module, string_name
    if key not in definition_cache:
        pool = [module]

        for n in filter(is_basenode, pool):
            pool += (n.children[1],) if n.type in {"classdef", "funcdef"} else n.children

        for n in filter(is_namenode, pool):
            if n.value == string_name and n.get_definition(include_setitem=True):
                break
        else:
            n = None
        definition_cache[key] = n
    return definition_cache[key]


def get_stub_module_context(self: dict, module_value: StubModuleValue):
    return self.setdefault(module_value, CachedStubModuleContext(module_value))

_contexts = instanced_default_cache(get_stub_module_context)


def get_stub_filter(self: dict, module_value: StubModuleValue):
    return self.setdefault(module_value, [CachedStubFilter(_contexts[module_value])])


def optimize_StubModules():
    stub_filter_cache = instanced_default_cache(get_stub_filter)

    def get_filters(self: StubModuleValue, origin_scope=None):
        yield from stub_filter_cache[self]

    StubModuleValue.get_filters = get_filters
    StubModuleContext.is_stub   = truthy_noargs
