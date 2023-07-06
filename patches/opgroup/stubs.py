# Implements stub optimizations.

from jedi.inference.gradual.stub_value import StubModuleValue, StubModuleContext, StubFilter, StubName
from jedi.inference.value.klass import ClassMixin, ClassValue

from textension.utils import instanced_default_cache, truthy_noargs, _named_index, _TupleBase
from ..common import _check_flows, find_definition, state_cache
from ..tools import is_basenode, is_namenode, state
from itertools import repeat


stubfilter_names_cache   = {}
module_definitions_cache = {}


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


def apply():
    optimize_StubModules()
    optimize_ClassMixin_get_filters()


class DeferredStubName(_TupleBase, StubName):
    parent_context = _named_index(0)
    tree_name      = _named_index(1)


@state_cache
def get_scope_name_strings(scope):
    namedefs = []
    pool  = scope.children[:]

    for n in filter(is_basenode, pool):
        if n.type in {"classdef", "funcdef"}:
            # Add straight to ``namedefs`` we know they are definitions.
            namedefs += [n.children[1].value]

        elif n.type == "simple_stmt":
            n = n.children[0]

            if n.type == "expr_stmt":
                name = n.children[0]
                # Could be ``atom_expr``, as in dotted name. Skip those.
                if name.type == "name":
                    namedefs += [name.value]
                else:
                    print(name.type)

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
                                namedefs += [alias.value]

            elif n.type == "atom":
                # ``atom`` nodes are too ambiguous to extract names from.
                # Just into the pool and look for names the old way.
                pool += n.children

        elif n.type == "decorated":
            pool += [n.children[1]]
        else:
            pool += n.children

    # Get name definitions.
    for n in filter(is_namenode, pool):
        if n.get_definition(include_setitem=True):
            namedefs += [n.value]

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
        pool += [n.children[1]] if n.type in {"classdef", "funcdef"} else n.children

    for n in filter(is_namenode, pool):
        if n.value in module_names:
            names += [n]

    ret = stub_filter._filter(names)
    ret = self[stub_filter] = list(map(DeferredStubName, zip(repeat(context), ret)))
    return ret


stub_values_cache = instanced_default_cache(get_stub_values)


class CachedStubFilter(StubFilter):
    _parso_cache_node = None
    _check_flows = _check_flows

    def __init__(self, parent_context):
        self._until_position = None
        self._origin_scope   = None
        self.parent_context  = parent_context
        self._node_context   = parent_context
        self._parser_scope   = parent_context.tree_node
        self._used_names     = parent_context.tree_node.get_root_node().get_used_names()
        self.cache = {}

    def get(self, name):
        if name := get_module_definition_by_name(self._parser_scope, name):
            names = self._convert_names((name,))
            return names
        return []

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
            pool += [n.children[1]] if n.type in {"classdef", "funcdef"} else n.children

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


get_filters_orig = ClassValue.get_filters


def get_class_filters(self, key):
    tree_node, module_value, kw = key
    value = ClassValue(state, _contexts[module_value], tree_node)
    result = list(get_filters_orig(value, **dict(kw)))
    self[key] = result
    return result


def optimize_ClassMixin_get_filters():
    stub_class_filter_cache = instanced_default_cache(get_class_filters)

    def get_filters(self: ClassValue, **kw):
        # XXX: The type check is needed because generics, which have custom
        # data, are being passed as ClassValues.
        if self.parent_context.is_stub() and type(self) is ClassValue:
            module_value = self.parent_context.get_root_context()._value
            return stub_class_filter_cache[self.tree_node, module_value, tuple(kw.items())]
        return get_filters_orig(self, **kw)

    ClassMixin.get_filters = get_filters