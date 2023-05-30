# This adds optimizations for used names lookup and filtering.

from parso.python.tree import Module, Name, BaseNode, Leaf
from collections import defaultdict
from textension.utils import _patch_function


def apply():
    # Prerequisite.
    BaseNode.is_nonterminal = True
    Leaf.is_nonterminal = False

    optimize_name_get_definition()
    optimize_get_definition_names()  # XXX: Requires optimize_name_get_definition
    optimize_Module_get_used_names()  # XXX: Requires prepare_optimizations and optimize_name_get_definition.
    optimize_AbstractUsedNamesFilter_get_and_values()  # XXX Requires UsedNames and optimize_name_get_definition
    optimize_ParserTreeFilter_filter()


# Optimizes ParserTreeFilter._filter.
# XXX: This is not a drop-in replacement. It only works in conjunction with lookup optimizations.
# Names are cached on the filter.
# The filter is cached on the node.
def optimize_ParserTreeFilter_filter():
    from jedi.inference.filters import ParserTreeFilter, flow_analysis
    from jedi.inference.flow_analysis import reachability_check
    from jedi.inference.gradual.stub_value import StubFilter
    from jedi.parser_utils import get_parent_scope
    from itertools import compress, takewhile, repeat
    from builtins import map, filter, sorted, hash, tuple
    from operator import attrgetter

    get_start = attrgetter("start_pos")
    is_stubfilter = StubFilter.__instancecheck__
    not_unreachable = flow_analysis.UNREACHABLE.__ne__
    not_reachable = flow_analysis.REACHABLE.__ne__

    # Reachable cache for StubFilter
    stubfilter_cache = {}

    def exists_in_compiled_module(self: StubFilter, value: str):
        try:
            cache = stubfilter_cache[self]
        except:
            context = self.parent_context
            values = context._value.non_stub_value_set

            if context.is_compiled():
                # If the stub module has a compiled counterpart, use its __dict__.
                g = (k for v in values for k in v.access_handle.access._obj.__dict__.keys())
            else:
                g = (n.string_name for v in values for n in next(v.get_filters()).values())

            cache = stubfilter_cache[self] = set(g)
        return value in cache

    def stubfilter_post_filtering(self: StubFilter, names):
        for name in names:
            # Check for stub import aliasing.
            if definition := name.get_definition():
                if definition.type in {"import_from", "import_name"}:
                    if name.parent.type not in {"import_as_name", "dotted_as_name"}:
                        continue

            # Simple heuristic to see if a name starts with an underscore.
            # If it does, we then check the keys in the compiled module.
            value = name.value
            if value[0] is "_" and (value[:2] != "__" != value[-2:]):
                if not exists_in_compiled_module(self, value):
                    continue
            yield name

    def _filter(self: ParserTreeFilter, names):
        # This inlines AbstractFilter._filter.
        if end := self._until_position:
            names = list(compress(names, map(end.__gt__, map(get_start, names))))

        # The scope which ``names`` is tested for reachability.
        scope = self._parser_scope

        tmp = []
        # Inlines ParserTreeFilter._is_name_reachable
        for name in names:

            # XXX: This code assumes ``get_used_names`` called ``get_parent_scope`` on the name.
            # if name.scope_cache[False] is scope:
            #     tmp += [name]
            #     continue

            parent_type = name.parent.type
            if parent_type != "trailer":
                base = name if parent_type not in {"classdef", "funcdef"} else name.parent
                try:
                    parent_scope = base.scope_cache[False]
                except:
                    parent_scope = get_parent_scope(base)
                if parent_scope == scope:
                    tmp += [name]
                continue

        if tmp:
            # StubFilter instances require extra reachable checks.
            if is_stubfilter(self):
                tmp = stubfilter_post_filtering(self, tmp)

            sorted_names = sorted(tmp, key=get_start, reverse=True)

            to_check = map(reachability_check,
                           repeat(self._node_context),
                           repeat(self._parser_scope),
                           sorted_names,
                           repeat(self._origin_scope))
            selectors = takewhile(not_reachable, map(not_unreachable, to_check))
            return compress(sorted_names, selectors)
        return []

    ParserTreeFilter._filter = _filter


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


class Collection(list):
    cached_data = {}

    # XXX: What is this for? Do we need this?
    def __getattr__(self, __name):
        if __name == "value":
            value = self[0].value
            setattr(self, "value", value)
            return value
        raise AttributeError
   
    # XXX: What is this for? Do we need this?
    def get_cached(self, filter_func):
        cached_data = self.cached_data
        comp = tuple(self)
 
        value = self.value
        try:
            prev_comp, dct = cached_data[value]
        except:
            dct = {}
            cached_data[value] = comp, dct
        else:
            if prev_comp == comp:
                try:
                    return dct[filter_func]
                except:
                    pass
        ret = dct[filter_func] = filter_func(self)
        return ret


# TODO: Implement a get(string) method. Seems jedi requires this for in some cases.
# TAG:  UsedNames
class UsedNames(defaultdict):
    module: Module
    scope_map: dict

    def get(self, *args):
        return ()

    def __init__(self, module):
        super().__init__(Collection)
        self.module = module

        # Update tagged nodes from the DiffParser.
        self.module.updates = set()

        self.module.names = set()
        self.scope_map = defaultdict(Collection)

        # A set of nodes at module level.
        self.nodes = set()

    def update_module(self):
        from builtins import map, set
        module = self.module

        all_names = module.names
        old_nodes = set()
        # old_nodes = self.nodes  # XXX: While debugging. This makes incremental updates no longer work.
        new_nodes = self.nodes = set(module.children)

        remove_nodes = old_nodes - new_nodes

        for branch in remove_nodes:
            all_names -= branch.names

        new_nodes -= old_nodes

        new_nodes |= module.updates
        module.updates.clear()

        scope_map = self.scope_map
        scope_map.default_factory = Collection

        is_namenode = Name.__instancecheck__
        is_basenode = BaseNode.__instancecheck__  # Means node has children.


        _get_definition = Name._get_definition
        new_module_names = []

        for branch in new_nodes - old_nodes:

            pool = [branch]
            branch_names = []

            # Process branch nodes.
            for node in filter(is_basenode, pool):
                pool += node.children

            # Process Name nodes in branch.
            for name in filter(is_namenode, pool):
                name.definition = {(False, True): _get_definition(name, False, True)}
                scope = get_parent_scope_fast(name)

                name.scope_cache = {False: scope}
                scope_map[scope, name.value] += [name]

                if scope is module:
                    new_module_names += [name]

                branch_names += [name]

            # The module and its direct decendants store their children in
            # sets to speed up ParserTreeFilter.filter by using exclusion sets.
            # XXX: Does this really have to be a set? Where is branch.names looked up?
            branch.names = set(branch_names)

        # TODO: Is this actually needed?
        all_names |= set(new_module_names)
        # TODO: Is this also actually needed? Shouldn't this be done on the scope map?
        scope_map.default_factory = None
        self.default_factory = None
        return self


# XXX: Requires prepare_optimizations.
# XXX: Requires optimize_name_get_definition.
# Optimize Module.get_used_names.
def optimize_Module_get_used_names():

    # XXX: Unmanaged, but these are modules.
    lookup_cache = {}

    def get_used_names(self: Module):
        # print(self, id(self))
        if (used_names := self._used_names) is not None:
            return used_names
        if self in lookup_cache:
            self._used_names = used_names = lookup_cache[self]
        else:
            self._used_names = used_names = lookup_cache[self] = UsedNames(self)
        return used_names.update_module()

    Module.get_used_names = get_used_names


# XXX: This optimization requires UsedNames, and UsedNames requires this!
# Optimizes _AbstractUsedNamesFilter.get to inline ``_get_definition_names``
# and to allow it query the filter's node context to get the exact scope.
# TAG: AbstractUsedNamesFilter
def optimize_AbstractUsedNamesFilter_get_and_values():
    # Requires ``Name._get_definition`` optimization.

    from jedi.inference.filters import _AbstractUsedNamesFilter
    from jedi.parser_utils import get_parent_scope
    from parso.python.tree import Name
    from itertools import repeat

    # Same as Name.get_definition, but without trying the cache.
    get_definition = Name.get_definition
    from dev_utils import ncalls, get_callsite_string

    def get(self: _AbstractUsedNamesFilter, name):
        node = self._node_context.tree_node
        scope_map = self._used_names.scope_map
        try:
            names = scope_map[node, name]
        except:
            print("can't find", name, get_callsite_string(depth=3))
            return []

        try:
            ret = names.cached_result
        except:  # Assume AttributeError.
            ret = names.cached_result = []  
            for name in names:
                definition = get_definition(name, include_setitem=True)
                name.definition = {(False, True): definition}
                if definition is not None:
                    ret += [name]
        return self._convert_names(self._filter(ret))

    def get(self: _AbstractUsedNamesFilter, name):
        node = self._node_context.tree_node

        ret = []

        while not (names := getattr(node, "names", ())):
            # print(f"Node '{node}' has no attribute 'names'. Asked for: {name}")
            if node := node.parent:
                continue
            break
            

        for n in names:
            if n.value == name:
                definition = get_definition(n, import_name_always=False, include_setitem=True)
                n.definition = {(False, True): definition}
                if definition is not None:
                    ret += [n]
        return self._convert_names(self._filter(ret))

    _AbstractUsedNamesFilter.get = get

    def values(self: _AbstractUsedNamesFilter):
        try:
            return self.cached_values
        except:
            # TODO: Defined names should be cached on the tree node.
            node = self._node_context.tree_node

            while not hasattr(node, "names"):
                node = get_parent_scope(node)

            ret = []
            names = node.names
            for name in names:
                if name.get_definition(include_setitem=True):
                    ret += [name]
            ret = list(self._filter(ret))
            ret = list(map(self.name_class, repeat(self.parent_context), ret))
            self.cached_values = ret
        return self.cached_values
    
    _AbstractUsedNamesFilter.values = values

    def _convert_names(self, names):
        # return map(self.name_class, repeat(self.parent_context), names)
        return list(map(self.name_class, repeat(self.parent_context), names))

    _AbstractUsedNamesFilter._convert_names = _convert_names


def optimize_name_get_definition():
    from parso.python.tree import Name, _GET_DEFINITION_TYPES, _IMPORTS

    # # This function takes a Name and returns the ancestor that defines it.
    # # If the name is the ``e`` part of ``except Exception as e``, then the
    # # ancestor that defines the try-clause (``try_stmt``) is returned.
    index = list.index

    # def _get_definition(self: Name, import_name_always=False, include_setitem=False):
    #     p: BaseNode = self.parent

    #     type = p.type
    #     if type in {"funcdef", "classdef", "except_clause"}:

    #         # ``funcdef`` or ``classdef``.
    #         children = p.children
    #         if self is children[1]:  # Same as ``self == parent.name``.
    #             return p 

    #         # self is ``e`` part of ``except X as e``.
    #         elif type == "except_clause":
    #             if children[index(children, self) - 1].value == "as":
    #                 return p.parent  # The try_stmt.
    #         return None

    #     while p:
    #         type = p.type
    #         if type not in _GET_DEFINITION_TYPES:
    #             p = p.parent
    #             continue

    #         elif self in p.get_defined_names(include_setitem):
    #             return p
    #         elif import_name_always and type in _IMPORTS:
    #             return p
    #         return None

    def _get_definition(self: Name, import_name_always=False, include_setitem=False):
        p: BaseNode = self.parent

        type       = p.type
        definition = None

        if type in {"funcdef", "classdef", "except_clause"}:

            # ``funcdef`` or ``classdef``.
            children = p.children
            if self is children[1]:  # Same as ``self == parent.name``.
                definition = p 

            # self is ``e`` part of ``except X as e``.
            elif type == "except_clause" and  children[index(children, self) - 1].value == "as":
                definition = p.parent  # The try_stmt.
        else:
            while p:
                type = p.type
                if type not in _GET_DEFINITION_TYPES:
                    p = p.parent
                    continue

                elif self in p.get_defined_names(include_setitem) or import_name_always and type in _IMPORTS:
                    definition = p
                break
        return definition

    # XXX: Requires __slots__ removed on BaseNode and ``definition`` to exist on the node.
    def get_definition(self: Name, import_name_always=False, include_setitem=False):
        args = import_name_always, include_setitem
        try:
            # Try directly. Definition is mainly done in module_update.
            return self.definition[args]
        except:
            definition = self.definition = getattr(self, "definition", {})

        ret = definition[args] = orig_get_definition(self, *args)
        return ret

    orig_get_definition  = Name.get_definition

    Name.get_definition  = get_definition
    Name._get_definition = _get_definition


# XXX: Requires optimize_name_get_definition.
# Optimizes _get_definition_names in 'jedi/inference/filters.py'
def optimize_get_definition_names():
    from parso.python.tree import Name
    from jedi.inference.filters import _get_definition_names


    # Same as Name.get_definition, but without trying the cache.
    get_definition = Name.get_definition

    # TAG: get_definition_names
    # The result is cached on the ``Collection`` object instead.

    def _get_definition_names_o(parso_cache_node, used_names: UsedNames, string):
        # Jedi sometimes looks up strings like ``__init__`` on classes that 
        # don't define them. In this case we just jedi sort it automagically.
        try:
            names = used_names.scope_map[parso_cache_node.node, string]
        except:
            # print("can't find", repr(string), "on", parso_cache_node.node, get_callsite_string(depth=3))  # Debug purposes.
            return []

        try:
            ret = names.cached_result
        except:  # Assume AttributeError.
            ret = names.cached_result = []

            for name in names:
                definition = get_definition(name, include_setitem=True)
                name.definition = {(False, True): definition}
                if definition is not None:
                    ret += [name]
        return ret

    _patch_function(_get_definition_names, _get_definition_names_o)
