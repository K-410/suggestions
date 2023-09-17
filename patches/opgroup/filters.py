# This module implements optimizations for various filters and stubs.
from jedi.inference.compiled.getattr_static import getattr_static
from jedi.inference.compiled.access import ALLOWED_DESCRIPTOR_ACCESS
from jedi.inference.gradual.stub_value import StubModuleValue, StubModuleContext, StubFilter
from jedi.inference.value.klass import ClassFilter

from textension.utils import instanced_default_cache, truthy_noargs, _named_index, Aggregation, lazy_overwrite, _forwarder, inline, Variadic, _variadic_index
from ..common import _check_flows, AggregateStubName, filter_basenodes, filter_names, filter_funcdefs, filter_params
from itertools import repeat


def apply():
    optimize_SelfAttributeFilter()
    optimize_ClassFilter()
    optimize_ParserTreeFilter()
    optimize_AnonymousMethodExecutionFilter()
    optimize_CompiledValueFilter()
    optimize_CompiledValue_get_filters()
    optimize_BaseTreeInstance_get_filters()
    optimize_ClassMixin_get_filters()

    optimize_StubModules()
    optimize_AbstractUsedNamesFilter_convert_names()
    optimize_AbstractUsedNamesFilter_get()


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

    def __repr__(self):
        return f"<AggregateClassFilter: {self.parent_context}"


def is_allowed_getattr(obj, name):
    try:
        attr, is_get_descriptor = getattr_static(obj, name)
    except AttributeError:
        return False, False
    else:
        if is_get_descriptor and type(attr) not in ALLOWED_DESCRIPTOR_ACCESS:
            return True, True
    return True, False


# Optimizes ``values`` to not access attributes off a compiled value when
# converting them into names. Instead they are read only when inferred.
def optimize_CompiledValueFilter():
    from jedi.inference.compiled.value import CompiledValueFilter
    from itertools import repeat
    from ..common import state_cache, AggregateCompiledName
    from builtins import dir

    def cached_dir(obj):
        try:
            return _cached_dir(obj)
        except TypeError:
            return dir(obj)

    _cached_dir = state_cache(dir)

    @state_cache
    def values(self: CompiledValueFilter):
        value = self.compiled_value
        sequences = zip(repeat(value), cached_dir(value.access_handle.access._obj))
        # NOTE: type completions is not added, because why would we on a compiled value?
        return list(map(AggregateCompiledName, sequences))

    CompiledValueFilter.values = values


# Optimizes SelfAttributeFilter.values to exclude stubs and limit the search
# for self-assigned definitions to within a class' methods.
def optimize_SelfAttributeFilter():
    from jedi.inference.value.instance import SelfAttributeFilter
    from parso.python.tree import Class
    from ..common import AggregateSelfName

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

                for n in filter_funcdefs(class_nodes):
                    pool += n.children[-1].children

                # Recurse node trees.
                for n in filter_basenodes(pool):
                    pool += n.children

                # Get name definitions.
                for n in filter_names(pool):
                    if n.get_definition(include_setitem=True):
                        names += n,

                names = self._filter(names)
                return self._convert_names(names)
        return []

    SelfAttributeFilter.values = values

    def _convert_names(self: SelfAttributeFilter, names):
        data = zip(repeat(self._instance), repeat(self._node_context), names)
        return list(map(AggregateSelfName, data))

    SelfAttributeFilter._convert_names = _convert_names


def optimize_ClassFilter():
    from jedi.inference.value.klass import ClassFilter
    from parso.python.tree import Class
    from builtins import list, map, zip
    from itertools import repeat
    from ..common import AggregateClassName, get_scope_name_definitions
    from ..common import get_cached_scope_definitions

    stub_cache = {}

    def values(self: ClassFilter):
        scope = self._parser_scope
        names = []

        # Fastest way to avoid adding a class' arglist (bases) names.
        if scope.__class__ is Class:
            scope = scope.children[-1]

        if self._context_value.is_stub():
            if scope not in stub_cache:
                stub_cache[scope] = self._filter(get_scope_name_definitions(scope))
            names = stub_cache[scope]
        else:
            names = get_scope_name_definitions(scope)
        return self._convert_names(names)

    def _filter(self: ClassFilter, names):
        tmp = []
        for name in names:
            # TODO: Use functional style.
            if name.value.startswith("__") and not name.value.endswith("__"):
                continue
            tmp += name,
        return tmp

    def get(self: ClassFilter, name_str: str):
        names = get_cached_scope_definitions(self._parser_scope)[name_str]
        data = zip(repeat(self._class_value),
                   names,
                   repeat(self._node_context),
                   repeat(not self._is_instance))
        return list(map(AggregateClassName, data))
        return self._convert_names(names)

    def _convert_names(self: ClassFilter, names):
        data = zip(repeat(self._class_value),
                   names,
                   repeat(self._node_context),
                   repeat(not self._is_instance))
        return list(map(AggregateClassName, data))

    # ClassFilter.get = get
    # ClassFilter._filter = _filter
    ClassFilter.values = values
    ClassFilter._check_flows = _check_flows
    ClassFilter._convert_names = _convert_names


def optimize_ParserTreeFilter():
    from jedi.inference.filters import ParserTreeFilter
    from textension.utils import starchain
    from itertools import repeat

    from ..common import get_cached_scope_definitions, AggregateTreeNameDefinition
    from ..common import get_scope_name_definitions, get_parent_scope_fast
    from ..common import _check_flows, filter_until

    # XXX: Unusable. Needs revisiting.
    def get(self: ParserTreeFilter, name_str: str):
        scope = self._parser_scope

        definitions = []
        while scope:
            if names := get_cached_scope_definitions(scope)[name_str]:
                definitions += names,
            scope = get_parent_scope_fast(scope)

        if definitions:
            definitions[0] = filter_until(self._until_position, definitions[0])
            return list(map(AggregateTreeNameDefinition, zip(repeat(self.parent_context), starchain(definitions))))
        return []

    def values(self: ParserTreeFilter):
        names = get_scope_name_definitions(self._parser_scope)
        names = self._filter(names)
        return self._convert_names(names)
    
    def _filter(self: ParserTreeFilter, names):
        names = filter_until(self._until_position, names)

        # XXX: Needed because jedi still calls this with names from all over.
        names = [n for n in names if self._is_name_reachable(n)]
        return _check_flows(self, names)

    # ParserTreeFilter.get = get
    ParserTreeFilter.values = values
    ParserTreeFilter._filter = _filter
    ParserTreeFilter._check_flows = _check_flows


def optimize_AnonymousMethodExecutionFilter():
    from jedi.inference.value.instance import AnonymousMethodExecutionFilter
    from parso.python.tree import Lambda
    from ..common import get_cached_scope_definitions
    from ..common import get_scope_name_definitions

    def get(self: AnonymousMethodExecutionFilter, name_string):
        names = get_cached_scope_definitions(self._parser_scope)[name_string]
        names = self._filter(names)
        return self._convert_names(names)

    def values(self: AnonymousMethodExecutionFilter):
        names = []
        scope = self._parser_scope
        children = scope.children

        # Lambda nodes have no parentheses that delimit parameters.
        if scope.__class__ is Lambda:
            params = children[1]
            if params.type == "operator":  # The colon.
                params = []
            elif params.type == "param":
                params = [params]
        else:
            params = children[2].children

        # Get parameter name definitions.
        for param in filter_params(params):
            names += param.children[0],

        # The suite.
        names += get_scope_name_definitions(children[-1])
        names = self._filter(names)
        return self._convert_names(names)
    
    AnonymousMethodExecutionFilter.get = get
    AnonymousMethodExecutionFilter.values = values
    AnonymousMethodExecutionFilter._check_flows = _check_flows


def optimize_CompiledValue_get_filters():
    from jedi.inference.compiled.value import CompiledValue
    from ..common import AggregateCompiledValueFilter

    def get_filters(self: CompiledValue, is_instance=False, origin_scope=None):
        yield AggregateCompiledValueFilter((self, is_instance))

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

        if not cached_type_filter:
            instance, = cached_builtins.type.py__call__(NoArguments)
            for f in instance.class_value.get_filters(origin_scope=None, is_instance=True):
                cached_type_filter = InstanceClassFilter(instance, f)
                break
        return cached_type_filter

    @state_cache
    def get_cached_compiled_class_filters(cls: ClassValue, is_instance: bool):
        return list(cls.get_filters(is_instance=is_instance))

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
            yield from stub_class_filter_cache[key]
        else:
            yield from _get_filters(self, **kw)

    ClassMixin.get_filters = get_filters



@inline
def get_versioned_nodes(if_stmt):
    from parso.python.tree import String
    from functools import partial
    from itertools import count
    from builtins import tuple
    import operator
    from ..common import filter_keywords, filter_numbers, map_values, filter_node_type
    from sys import platform, version_info

    logic_map = {
        ">":  operator.gt,
        ">=": operator.ge,
        "==": operator.eq,
        "!=": operator.ne,
        "<":  operator.lt,
        "<=": operator.le,
    }

    tests = {"or_test":  any, "and_test": all}

    def test_node(n):
        if n.type == "comparison":
            lhs, op, rhs = n.children
            string = lhs.get_code()
            if rhs.__class__ is String and "sys.platform" in string:
                return logic_map[op.value](platform, rhs.value.strip("\"\'"))

            elif "sys.version_info" in string and \
                (n := next(filter_node_type("testlist_comp", rhs.children), None)):
                    ints = tuple(to_ints(map_values(filter_numbers(n.children))))
                    return logic_map[op.value](version_info, ints)

            assert False, "Should not be here when testing on typeshed stubs."
        return tests[n.type](map_test_node(n.children[::2]))
    
    @inline
    def map_test_node(nodes):
        return partial(map, test_node)
    @inline
    def to_ints(number_strings):
        return partial(map, int)

    def get_versioned_nodes(if_stmt):
        # 0    1    2    3    4     5    6    7    8     9    10
        # if test   :  suite elif test   :  suite else   :  suite
        children = if_stmt.children
        for index, keyword in zip(count(step=4), filter_keywords(children)):
            if keyword.value == "else":
                return children[index + 2].children

            # ``if`` and ``elif``.
            elif test_node(children[index + 1]):
                return children[index + 3].children
        return ()
    return get_versioned_nodes


def get_stub_values(self: dict, stub_filter: "CachedStubFilter"):
    context = stub_filter.parent_context

    scope = context._value.tree_node

    namedefs = []
    pool  = scope.children[:]

    for n in filter_basenodes(pool):
        if n.type in {"classdef", "funcdef"}:
            # Add straight to ``namedefs`` we know they are definitions.
            namedefs += n.children[1],

        elif n.type == "simple_stmt":
            n = n.children[0]

            if n.type == "expr_stmt":
                op = n.children[1]
                # ``n.children[0].type`` must now be "name".

                if op.type == "annassign":
                    op = op.children[0]

                # ``op.type`` must now be "operator".

                # Only assignments/annotations without a single leading
                # underscore are exported by stubs per PEP 484.
                if op.value in {"=", ":"}:
                    namedefs += n.children[0],

            elif n.type == "import_from":
                name = n.children[3]
                if name.type == "operator":
                    # Starred imports are ignored in stubs.
                    if name.value == "*":
                        continue
                    name = n.children[4]

                # Only matching aliased imports are exported for stubs.
                if name.type == "import_as_names":
                    for name in name.children[::2]:
                        if name.type == "import_as_name":
                            name, alias = name.children[::2]
                            if name.value == alias.value:
                                namedefs += alias,

            elif n.type == "atom":
                # ``atom`` nodes are too ambiguous to extract names from.
                # Just into the pool and look for names the old way.
                pool += n.children

        elif n.type == "decorated":
            # Omit decorated classes using ``type_check_only``.
            for dec in iter(decorators := [n.children[0]]):
                if dec.type == "decorators":
                    decorators += dec.children
                elif dec.children[1].value == "type_check_only":
                    break
            else:
                pool += n.children[1],

        # Python version/platform-dependent definitions.
        elif n.type == "if_stmt":
            pool += get_versioned_nodes(n)

        else:
            pool += n.children

    tmp = []
    for name in reversed(namedefs):
        value = name.value
        if value[0] is not "_" or value[-1] is "_":
            tmp += name,

    ret = self[stub_filter] = stub_filter._convert_names(stub_filter._filter(tmp))
    return ret


stub_values_cache = instanced_default_cache(get_stub_values)


class CachedStubFilter(Variadic, StubFilter):
    _until_position   = None
    _origin_scope     = None
    _parso_cache_node = None

    parent_context    = _variadic_index(0)
    _parser_scope     = _forwarder("parent_context.tree_node")
    _node_context     = _forwarder("parent_context")

    _check_flows = _check_flows

    @lazy_overwrite
    def _used_names(self):
        return self.parent_context.tree_node.get_root_node().get_used_names()

    def get(self, name):
        if name := get_module_definition_by_name(self._parser_scope, name):
            return (AggregateStubName((self.parent_context, name)),)
        return ()

    def _filter(self, names):
        return self._check_flows(names)

    def values(self: StubFilter):
        return stub_values_cache[self]


class CachedStubModuleContext(StubModuleContext):
    get_filters = _forwarder("_value.get_filters")


definition_cache = {}


def get_module_definition_by_name(module, string_name):
    key = module, string_name
    if key not in definition_cache:
        pool = [module]

        for n in filter_basenodes(pool):
            pool += (n.children[1],) if n.type in {"classdef", "funcdef"} else n.children

        for n in filter_names(pool):
            if n.value == string_name and n.get_definition(include_setitem=True):
                break
        else:
            n = None
        definition_cache[key] = n
    return definition_cache[key]


def optimize_StubModules():
    @instanced_default_cache
    def stub_contexts(self: dict, module_value: StubModuleValue):
        return self.setdefault(module_value, CachedStubModuleContext(module_value))

    @instanced_default_cache
    def stub_filters(self: dict, module_value: StubModuleValue):
        return self.setdefault(module_value, [CachedStubFilter(stub_contexts[module_value])])

    def get_filters(self: StubModuleValue, *_, **kw):
        yield from stub_filters[self]

    StubModuleValue.get_filters = get_filters
    StubModuleContext.is_stub   = truthy_noargs


def optimize_AbstractUsedNamesFilter_convert_names():
    from jedi.inference.gradual.stub_value import StubFilter
    from jedi.inference.filters import _AbstractUsedNamesFilter
    from jedi.api.interpreter import MixedParserTreeFilter
    from ..common import AggregateTreeNameDefinition

    _AbstractUsedNamesFilter.name_class = AggregateTreeNameDefinition

    # XXX: Technically not correct. Should be AggregateMixedTreeName, but at
    # no point has the superclass infer never returned a value.
    MixedParserTreeFilter.name_class = AggregateTreeNameDefinition
    StubFilter.name_class = AggregateStubName

    def _convert_names(self: _AbstractUsedNamesFilter, names):
        return list(map(self.name_class, zip(repeat(self.parent_context), names)))

    _AbstractUsedNamesFilter._convert_names = _convert_names


def optimize_AbstractUsedNamesFilter_get():
    from jedi.inference.filters import _AbstractUsedNamesFilter
    from ..common import get_cached_scope_definitions

    def get(self: _AbstractUsedNamesFilter, name):
        names = get_cached_scope_definitions(self._parser_scope)[name]
        names = self._filter(names)
        return self._convert_names(names)

    _AbstractUsedNamesFilter.get = get
