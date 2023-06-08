from jedi.inference.value.function import BaseFunctionExecutionContext
from jedi.inference.context import ModuleContext
from parso.python.tree import Name
from parso.tree import search_ancestor
from itertools import chain

def apply():
    optimize_AbstractContext()



# Optimizes AbstractContext to infer names based on the queried name node
# instead of its string value.
def optimize_AbstractContext():
    from jedi.inference.base_value import ValueSet
    from jedi.inference.context import AbstractContext, _get_global_filters_for_name
    from jedi.inference.filters import MergedFilter, ParserTreeFilter
    from jedi.inference.finder import _remove_del_stmt
    from ..tools import state
    from itertools import repeat

    py__getattribute__orig = AbstractContext.py__getattribute__

    def py__getattribute__(self: AbstractContext, name_or_str, name_context=None,
                           position=None, analysis_errors=True):

        if isinstance(name_or_str, Name):
            node = go_to_definition(self, name_or_str, position)
            if values := node and node.infer():
                return values
        return py__getattribute__orig(self, name_or_str, name_context, position, analysis_errors)

    AbstractContext.py__getattribute__ = py__getattribute__

    def adjust_position(name, position):
        if position:
            n = name
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
        return position
    
    from parso.python.tree import _GET_DEFINITION_TYPES
    from parso.python.tree import Name, BaseNode
    is_basenode = BaseNode.__instancecheck__
    is_namenode = Name.__instancecheck__
    from jedi.inference.names import TreeNameDefinition

    def go_to_definition(context, ref, position):

        namedef = get_definition(ref, include_setitem=True)
        if namedef:
            return TreeNameDefinition(context, namedef)

        else:
            pass
        # Position adjustment from _get_global_filters_for_name.
        position = adjust_position(ref, position)
        module = ref.get_root_node()
        ref_scope = get_parent_scope_fast(ref)

        p = ref
        while p := p.parent:
            ptype = p.type

            # We're on the dot operator.
            if ptype == "error_node":
                continue

            for name in filter(is_namenode, p.children):
                if name.value == ref.value:
                    if name is not ref:
                        return TreeNameDefinition(context, name)
            if ptype in {"for_stmt"}:
                for name in filter(is_namenode, p.children):
                    if name.value == ref.value and name is not ref:
                        return TreeNameDefinition(context, name)
                    pass
                pass
            for name in module.get_used_names()[ref.value]:
                if name.get_definition():
                    if get_parent_scope_fast(name) is p:
                        if name.start_pos < position:
                            return TreeNameDefinition(context, name)


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

                elif either or scope.children[-2].start_pos < node.start_pos:
                    break
            scope = scope.parent
    return scope


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

def get_definition(ref: Name, import_name_always=False, include_setitem=False):
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
            for n in p.get_defined_names(include_setitem):
                if value == n.value:
                    return n
                # or import_name_always and p.type in _IMPORTS:
                # return p
        elif p.type == "file_input":
            for n in get_module_definitions(p):
                if value == n.value:
                    return n
        p = p.parent


from parso.python.tree import Module, Name, BaseNode, Leaf

is_basenode = BaseNode.__instancecheck__  # Means node has children.
is_namenode = Name.__instancecheck__

module_definitions_cache = {}

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