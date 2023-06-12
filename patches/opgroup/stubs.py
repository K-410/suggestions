from jedi.inference.gradual.stub_value import StubModuleValue, StubModuleContext, StubFilter

from textension.utils import instanced_default_cache, truthy_noargs
from jedi.inference.flow_analysis import reachability_check, UNREACHABLE

from itertools import repeat
from ..tools import is_basenode, is_namenode

from sys import modules as _sys_modules
from types import ModuleType
from jedi.inference.value.module import ModuleValue
from parso.python.tree import Name

from parso.python.tree import Name, BaseNode
from jedi.inference.names import TreeNameDefinition


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

def _get_imported_module(module_name):
    if real_module := _sys_modules.get(module_name):
        if isinstance(real_module, ModuleType):
            return real_module
    return None


def on_missing_module_names(self: dict, stub):
    
    # If a real module exists and has already been imported, use that.
    if real_module := _get_imported_module(stub.name.string_name):
        keys = set(real_module.__dict__)

    else:
        keys = set()
        add_key = keys.add
        pool = []

        for f in stub._get_stub_filters(origin_scope=None):
            scope = f._parser_scope
            pool  += scope.children
            break

        for n in filter(is_basenode, pool):
            pool += [n.children[1]] if n.type in {"classdef", "funcdef"} else n.children

        # Get name definitions.
        for n in filter(is_namenode, pool):
            value = n.value
            if value in keys:
                continue

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

                # There's no guidelines for names with a single underscore. The
                # trend is to exclude these from stubs, but this means names like
                # ``sys._getframe`` won't be available.
                if value[0] is "_" != value[:2][-1]:
                    continue
                add_key(value)
    self[stub] = keys
    return keys


module_names_cache = instanced_default_cache(on_missing_module_names)
from operator import attrgetter
get_start_pos = attrgetter("line", "column")

# TAG: get_parent_scope_fast
# This version doesn't include flow checks nor read or write to cache.
def get_parent_scope_fast(node):
    if scope := node.parent:
        pt = scope.type

        either = (pt == 'tfpdef' and scope.children[0] == node) or \
                 (pt == 'param'  and scope.name == node)

        while True:
            stype = scope.type
            if stype in {"classdef", "funcdef", "lambdef", "file_input", "sync_comp_for", "comp_for"}:

                if stype in {"file_input", "sync_comp_for", "comp_for"}:
                    if stype != "comp_for" or scope.children[1].type != "sync_comp_for":
                        break

                elif either or get_start_pos(scope.children[-2]) < node.start_pos:
                    break
            scope = scope.parent
    return scope

class CachedStubFilter(StubFilter):
    _parso_cache_node = None

    def __init__(self, parent_context):
        self._until_position = None
        self._origin_scope   = None
        self.parent_context  = parent_context
        self._node_context   = parent_context
        self._parser_scope   = parent_context.tree_node
        self._used_names = parent_context.tree_node.get_root_node().get_used_names()
        self.cache = {}

    def get(self, name):
        try:
            return self.cache[name]
        except KeyError:
            return self.cache.setdefault(name, list(super().get(name)))

    def _check_flows(self, names):
        for name in sorted(names, key=lambda name: name.start_pos, reverse=True):
            check = reachability_check(
                context=self._node_context,
                value_scope=self._parser_scope,
                node=name,
                origin_scope=self._origin_scope
            )
            if check is not UNREACHABLE:
                yield name

    def get(self, name):
        if names := self._used_names.get(name, []):
            names = self._filter(filter(Name.get_definition, names))
            names = self._convert_names(names)
        return names

    # This check is different from how jedi implements it.
    def _is_name_reachable(self, name):
        parent = name.parent
        if parent.type == 'trailer':
            return False
        base_node = parent if parent.type in ('classdef', 'funcdef') else name
        return get_parent_scope_fast(base_node) is self._parser_scope

    def values(self: StubFilter):
        context = self.parent_context
        assert context.is_stub()
        value = self.parent_context._value
        assert isinstance(value, ModuleValue)
        module_names = module_names_cache[value]

        names = []
        pool = [value.tree_node]
        assert value.is_stub()

        for n in filter(is_basenode, pool):
            pool += [n.children[1]] if n.type in {"classdef", "funcdef"} else n.children

        for n in filter(is_namenode, pool):
            if n.value in module_names:
                names += [n]

        ret = self._filter(names)
        return list(map(self.name_class, repeat(self.parent_context), ret))


class CachedStubModuleContext(StubModuleContext):
    def get_filters(self, until_position=None, origin_scope=None):
        return list(self._value.get_filters())

    def py__getattribute__(self, name_or_str, name_context=None, position=None, analysis_errors=True):
        if namedef := find_definition(self, name_or_str, position):
            return namedef.infer()
        print("Stub py__getattribute__ failed for", name_or_str)
        return super().py__getattribute__(name_or_str, name_context, position, analysis_errors)


def get_definition(ref: Name):
    p = ref.parent
    type  = p.type
    value = ref.value

    if type in {"funcdef", "classdef", "except_clause"}:

        # self is the class or function name.
        children = p.children
        if value == children[1].value:  # Is the function/class name definition.
            return children[1]

        # self is the e part of ``except X as e``.
        elif type == "except_clause" and value == children[-1].value:
            return children[-1]

    while p:
        if p.type in definition_types:
            for n in p.get_defined_names(True):
                if value == n.value:
                    return n
        elif p.type == "file_input":
            for n in get_module_definitions(p):
                if value == n.value:
                    return n
        p = p.parent


def get_module_definitions(module):
    if module not in module_definitions_cache or \
       module_definitions_cache[module][0] != id(module._used_names):
        
        definitions = []
        module_definitions_cache[module] = (id(module._used_names), definitions)

        pool = [module]

        for n in filter(is_basenode, pool):
            pool += [n.children[1]] if n.type in {"classdef", "funcdef"} else n.children

        for n in filter(is_namenode, pool):
            if n.get_definition(include_setitem=True):
                definitions += [n]
    return module_definitions_cache[module][1]


def find_definition(context, ref, position):
    if namedef := get_definition(ref):
        return TreeNameDefinition(context, namedef)

    # Inlined position adjustment from _get_global_filters_for_name.
    if position:
        n = ref
        lambdef = None
        while n := n.parent:
            if n.type not in {"classdef", "funcdef", "lambdef"}:
                continue
            elif n.type == "lambdef":
                lambdef = n
            elif position < n.children[-2].start_pos:
                if not lambdef or position < lambdef.children[-2].start_pos:
                    position = n.start_pos
                break

    p = ref
    while p := p.parent:

        # We're on the dot operator.
        if p.type == "error_node":
            continue

        for name in filter(is_namenode, p.children):
            if name.value == ref.value and name is not ref:
                return TreeNameDefinition(context, name)
    return None


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
