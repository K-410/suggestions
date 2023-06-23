# Implements stub optimizations.

from jedi.inference.gradual.stub_value import StubModuleValue, StubModuleContext, StubFilter

from textension.utils import instanced_default_cache, truthy_noargs
from ..common import _check_flows, DeferredStubName, find_definition
from ..tools import is_basenode, is_namenode

from sys import modules as _sys_modules

from itertools import repeat
from operator import attrgetter
from types import ModuleType

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


def get_module_names(stub_module):
    # If the real counterpart exists for a stub and has already been
    # imported, we use its keys for names instead of parsing the stub.
    real_module = _sys_modules.get(stub_module.name.string_name)
    if isinstance(real_module, ModuleType):
        keys = set(real_module.__dict__)

    else:
        keys = set()
        add_key = keys.add
        pool = []

        for f in stub_module._get_stub_filters(origin_scope=None):
            scope = f._parser_scope
            pool += scope.children
            break

        for n in filter(is_basenode, pool):
            pool += [n.children[1]] if n.type in {"classdef", "funcdef"} else n.children

        # Get name definitions.
        for n in filter(is_namenode, pool):
            value = n.value
            if value not in keys:
                if definition := n.get_definition(include_setitem=True):
                    if definition.type in {"import_from", "import_name"}:
                        parent = n.parent

                        # Non-alias imports are not allowed to be exported.
                        if parent.type not in {"import_as_name", "dotted_as_name"}:
                            continue
                        # Aliased imports allowed only if it matches the name.
                        name, alias_name = parent.children[::2]
                        if name.value != alias_name.value:
                            continue
                    # Exclude names with a single underscore.
                    if value[0] is "_" != value[:2][-1]:
                        continue

                    add_key(value)
    return keys


def get_stub_values(self: dict, stub_filter: "CachedStubFilter"):
    context = stub_filter.parent_context
    value   = context._value
    module_names = get_module_names(value)

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

    def py__getattribute__(self, name_or_str, name_context=None, position=None, analysis_errors=True):
        if namedef := find_definition(self, name_or_str, position):
            return namedef.infer()
        ret = super().py__getattribute__(name_or_str, name_context, position, analysis_errors)
        if ret:
            print("CachedStubModuleContext.py__getattribute__ failed for", name_or_str, "but found on super()")
        else:
            print("CachedStubModuleContext.py__getattribute__ failed for", name_or_str, "and super() failed")
        return ret


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


def on_missing_stub_filter(self: dict, module: StubModuleValue):
    context = CachedStubModuleContext(module)

    self[module] = filters = [
        CachedStubFilter(context),
        # *module.iter_star_filters(),
        # DictFilter(module.sub_modules_dict()),
        # DictFilter(module._module_attributes_dict())
    ]
    return filters


def optimize_StubModules():
    stub_filter_cache = instanced_default_cache(on_missing_stub_filter)

    def get_filters(self: StubModuleValue, origin_scope=None):
        yield from stub_filter_cache[self]

    StubModuleValue.get_filters = get_filters
    StubModuleContext.is_stub   = truthy_noargs
