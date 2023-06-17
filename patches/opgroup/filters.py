# This module implements optimizations for various filter types.
from jedi.inference.compiled.getattr_static import getattr_static
from jedi.inference.compiled.access import ALLOWED_DESCRIPTOR_ACCESS
from jedi.inference.compiled.value import CompiledName, ValueSet
from jedi.inference.value.klass import ClassFilter

from itertools import repeat
from operator import attrgetter

from textension.utils import _named_index
from ..common import _check_flows, state_cache
from ..tools import is_basenode, is_namenode, state


def apply():
    optimize_SelfAttributeFilter_values()
    optimize_ClassFilter_values()
    optimize_AnonymousMethodExecutionFilter()
    optimize_ParserTreeFilter_values()
    optimize_CompiledValueFilter_values()


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

    parent_context = _named_index(0)  # parent_value.as_context()
    _parent_value  = _named_index(1)  # parent_value
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

        if isinstance(obj, type) and obj is not type:
            for filter in builtin_from_name(self._inference_state, 'type').get_filters():
                names = chain(names, filter.values())
        return names

    CompiledValueFilter.values = values


# Optimizes SelfAttributeFilter.values to exclude stubs and limit the search
# for self-assigned definitions to within a class' methods.
def optimize_SelfAttributeFilter_values():
    from jedi.inference.value.instance import SelfAttributeFilter
    from ..tools import is_basenode, is_namenode, is_funcdef, is_classdef
    from itertools import repeat

    def values(self: SelfAttributeFilter):
        names   = []
        scope   = self._parser_scope

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


@state_cache
def get_scope_name_definitions(scope):
    names = []
    pool  = scope.children[:]

    for n in filter(is_basenode, pool):
        pool += [n.children[1]] if n.type in {"classdef", "funcdef"} else n.children

    # Get name definitions.
    for n in filter(is_namenode, pool):
        if n.get_definition(include_setitem=True):
            names += [n]

    return names


def optimize_ClassFilter_values():
    from ..tools import is_classdef
    from ..common import DeferredDefinition

    stub_classdef_cache = {}

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
                names = self._filter(get_scope_name_definitions(scope))

        return list(map(DeferredDefinition, zip(repeat(context), names)))

    ClassFilter.values = values
    ClassFilter._check_flows = _check_flows


def optimize_AnonymousMethodExecutionFilter():
    from jedi.inference.value.instance import AnonymousMethodExecutionFilter

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

    def values(self: ParserTreeFilter):
        scope   = self._parser_scope

        names = get_scope_name_definitions(scope)
        names = self._filter(names)
        return self._convert_names(names)
    
    ParserTreeFilter.values = values
    ParserTreeFilter._check_flows = _check_flows


def optimize_ParserTreeFilter_filter():
    from jedi.inference.filters import flow_analysis, ParserTreeFilter
    from itertools import compress, takewhile, repeat
    from builtins import map
    from ..common import trace_flow

    get_start = attrgetter("start_pos")

    not_unreachable = flow_analysis.UNREACHABLE.__ne__
    not_reachable   = flow_analysis.REACHABLE.__ne__

    def _filter(self: ParserTreeFilter, names):
        if end := self._until_position:
            names = compress(names, map(end.__gt__, map(get_start, names)))

        to_check  = map(trace_flow, names, repeat(self._origin_scope))
        selectors = takewhile(not_reachable, map(not_unreachable, to_check))
        return compress(names, selectors)

    ParserTreeFilter._filter = _filter
