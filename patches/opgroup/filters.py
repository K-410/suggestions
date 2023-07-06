# This module implements optimizations for various filter types.
from jedi.inference.compiled.getattr_static import getattr_static
from jedi.inference.compiled.access import ALLOWED_DESCRIPTOR_ACCESS
from jedi.inference.compiled.value import CompiledName, ValueSet

from itertools import repeat
from operator import attrgetter

from textension.utils import _named_index
from ..common import _check_flows
from ..tools import is_basenode, state


def apply():
    optimize_SelfAttributeFilter_values()
    optimize_ClassFilter_values()
    optimize_ClassFilter_filter()
    optimize_AnonymousMethodExecutionFilter()
    optimize_ParserTreeFilter_values()
    optimize_CompiledValueFilter_values()
    optimize_AnonymousMethodExecutionFilter_values()
    optimize_ParserTreeFilter_filter()
    optimize_CompiledValue_get_filters()
    # optimize_ClassMixin_get_filters()  # XXX: Runs on startup. Bad.
    optimize_BaseTreeInstance_get_filters()


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
        return ValueSet([self.infer_compiled_value()])



# Optimizes ``values`` to not access attributes off a compiled value when
# converting them into names. Instead they are read only when inferred.
def optimize_CompiledValueFilter_values():
    from jedi.inference.compiled.value import CompiledValueFilter
    from itertools import repeat, chain

    def values(self: CompiledValueFilter):
        from jedi.inference.compiled import builtin_from_name
        names = []
        obj = self.compiled_value.access_handle.access._obj

        value = self.compiled_value
        sequences = zip(repeat(value), repeat(value.as_context()), dir(obj))
        names = map(DeferredCompiledName, sequences)
 
        # XXX: Why are we adding names to compiled values?
        # It doesn't make sense to add type completions if the object isn't a class.
        # if not self.is_instance and isinstance(obj, type) and obj is not type:
        #     for filter in builtin_from_name(self._inference_state, 'type').get_filters():
        #         names = chain(names, filter.values())
        return names

    CompiledValueFilter.values = values


# Optimizes SelfAttributeFilter.values to exclude stubs and limit the search
# for self-assigned definitions to within a class' methods.
def optimize_SelfAttributeFilter_values():
    from jedi.inference.value.instance import SelfAttributeFilter
    from ..tools import is_basenode, is_namenode, is_funcdef, is_classdef
    from itertools import repeat

    def values(self: SelfAttributeFilter):
        names = []
        scope = self._parser_scope

        # We only care about classes.
        if is_classdef(scope):
            context = self.parent_context

            # Stubs don't have self definitions.
            if not context.is_stub():
                class_nodes = scope.children[-1].children
                pool = []

                for n in filter(is_funcdef, class_nodes):
                    pool += n.children[-1].children

                # Recurse node trees.
                for n in filter(is_basenode, pool):
                    pool += n.children

                # Get name definitions.
                for n in filter(is_namenode, pool):
                    if n.get_definition(include_setitem=True):
                        names += [n]

                names = list(self._filter(names))
                names = list(map(self.name_class, repeat(context), names))
        return names

    SelfAttributeFilter.values = values


def get_function_name_definitions(function):
    
    namedefs = []
    pool  = function.children[:]
    for n in filter(is_basenode, pool):
        if n.type == "parameters":
            namedefs += n.children[1].children[::2]
        # pool += n.children

    return namedefs

def optimize_ClassFilter_values():
    from jedi.inference.value.klass import ClassFilter
    from ..common import DeferredDefinition, state_cache, get_scope_name_definitions
    from ..tools import is_classdef

    stub_classdef_cache = {}

    @state_cache
    def values(self: ClassFilter):
        context = self.parent_context
        scope   = self._parser_scope
        names   = []

        if is_classdef(scope):
            # The class suite.
            scope = scope.children[-1]

            # A user defined class' base can be a stub value, although the
            # context still is non-stub. So we check the node context also
            # to determine whether the class filter values can be cached.
            if context.is_stub() or self._node_context.is_stub():
                if scope not in stub_classdef_cache:
                    stub_classdef_cache[scope] = self._filter(get_scope_name_definitions(scope))
                names = stub_classdef_cache[scope]
            else:
                names = get_scope_name_definitions(scope)
                # names = self._filter(get_scope_name_definitions(scope))

        return list(map(DeferredDefinition, zip(repeat(context), names)))

    ClassFilter.values = values
    ClassFilter._check_flows = _check_flows


def optimize_ClassFilter_filter():
    from jedi.inference.value.klass import ClassFilter

    def _filter(self: ClassFilter, names):
        instanced = self._is_instance
        scope = self._parser_scope
        tmp = []

        for name in names:
            parent = name.parent
            parent_type = parent.type

            if parent_type in {"funcdef", "classdef"}:
                suite = parent.parent

                # For decorators inside classes.
                # XXX: Jedi still has some control of what gets passed to ``_filter``,
                # so we can't blindly assume names to be remotely in the same scope.
                if suite.type == "decorated":
                    while p := suite.parent:
                        if p is scope:
                            break
                        suite = p

                # Also check for overloaded/decorated functions.
                if suite.parent is scope:
                    tmp += [name]
                else:
                    pass

            elif parent_type == "expr_stmt":
                if parent.parent.parent.parent is scope:
                    # Either ``operator`` or ``annassign``.
                    # Annotations are assumed to exist on instances.
                    if parent.children[1].type == "operator" or instanced:
                        tmp += [name]
        return tmp

    ClassFilter._filter = _filter


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
            names += [param.children[0]]

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


def optimize_ClassMixin_get_filters():
    from jedi.inference.value.klass import ClassMixin
    from jedi.inference.value.klass import ClassFilter
    from jedi.inference.compiled import builtin_from_name
    from itertools import islice

    type_ = builtin_from_name(state, "type")
    type_values = []
    for instance in type_.py__call__(None):
        type_values += islice(instance.get_filters(), 2, 3)

    def get_filters(self: ClassMixin, origin_scope=None, is_instance=False, include_metaclasses=True, include_type_when_class=True):
        if include_metaclasses:
            if metaclasses := self.get_metaclasses():
                yield from self.get_metaclass_filters(metaclasses, is_instance)


        for cls in self.py__mro__():
            if cls.is_compiled():
                yield from cls.get_filters(is_instance=is_instance)
            else:
                yield ClassFilter(self, node_context=cls.as_context(), origin_scope=origin_scope, is_instance=is_instance)

        if not is_instance and include_type_when_class and self is not type_:
            yield from type_values

    ClassMixin.get_filters = get_filters


def optimize_CompiledValue_get_filters():
    from jedi.inference.compiled.value import CompiledValue, CompiledValueFilter
    from ..common import state, yield_filters_once

    # XXX: Disable decorator for now. Causes issues inside class bodies.
    # @yield_filters_once
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
