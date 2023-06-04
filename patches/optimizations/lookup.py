# This adds optimizations for used names lookup and filtering.

from parso.python.tree import Module, Name, BaseNode, Leaf
from collections import defaultdict
from textension.utils import _patch_function, factory


def apply():
    # Prerequisite.
    BaseNode.is_nonterminal = True
    Leaf.is_nonterminal = False

    optimize_name_get_definition()
    optimize_Module_get_used_names()    # XXX: Needs name_get_definition.
    optimize_AbstractUsedNamesFilter()  # XXX: Needs UsedNames and name_get_definition
    optimize_ParserTreeFilter_filter()
    optimize_GlobalNameFilter()

    optimize_get_parent_scope()


# XXX: Required because we filter differently now.
def optimize_GlobalNameFilter():
    from jedi.inference.filters import GlobalNameFilter

    from operator import attrgetter
    from itertools import compress

    get_parent_type = attrgetter("parent.type")
    is_global_statement = "global_stmt".__eq__

    from dev_utils import ncalls
    def get(self: GlobalNameFilter, name):
        try:
            names = self._used_names[name]
        except KeyError:
            return []
        return self._convert_names(self._filter(names))

    def _filter(self: GlobalNameFilter, names):
        raise Exception  # XXX: This should not be used. ``self.values()`` inlines this!
        ret = []
        for name in names:
            if name.parent.type == 'global_stmt':
                ret += [name]
        return ret

    def values(self: GlobalNameFilter):
        parent_types = map(get_parent_type, self._used_names)  # ``name.parent.type``
        selectors    = map(is_global_statement, parent_types)  # ... == "global_stmt"
        global_names = compress(self._used_names, selectors)
        # TODO: Inline self._convert_names and make functional?
        return self._convert_names(global_names)

    GlobalNameFilter.get = get
    GlobalNameFilter.values = values
    GlobalNameFilter._filter = _filter

# Optimizes ParserTreeFilter._filter.
# XXX: This is not a drop-in replacement. It only works in conjunction with lookup optimizations.
# Names are cached on the filter.
# The filter is cached on the node.
def optimize_ParserTreeFilter_filter():
    from jedi.inference.filters import flow_analysis, ParserTreeFilter
    from jedi.inference.flow_analysis import reachability_check
    from jedi.inference.gradual.stub_value import StubFilter
    from itertools import compress, takewhile, repeat
    from builtins import map, sorted
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

    from dev_utils import ncalls
    from jedi.inference.value.klass import ClassFilter

    def _filter(self: ParserTreeFilter, names):
        # This inlines AbstractFilter._filter.
        if end := self._until_position:
            names = compress(names, map(end.__gt__, map(get_start, names)))

        # The scope which ``names`` is tested for reachability.
        scope = self._parser_scope

        tmp = []
        # Inlines ParserTreeFilter._is_name_reachable
        for name in names:

            # XXX: This code assumes ``get_used_names`` called ``get_parent_scope`` on the name.
            if name.scope_cache[False] is scope:
                tmp += [name]
                continue

            parent_type = name.parent.type

            if parent_type != "trailer":
                if parent_type in {"classdef", "funcdef"}:
                    parent_scope = name.parent.scope_cache[False]
                else:
                    parent_scope = name.scope_cache[False]

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


# NOTE: Requires NodeOrLeaf to not use __slots__, or have "scope" added to it.
@factory
def map_parent_scopes(node):
    from parso.python.tree import Keyword
    from itertools import compress, count

    flows  = {"for_stmt", "if_stmt", "try_stmt", "while_stmt", "with_stmt"}

    scopes = {"classdef", "comp_for", "file_input", "funcdef",  "lambdef",
              "sync_comp_for"}

    any_scope = scopes | flows
    is_keyword = Keyword.__instancecheck__

    def map_parent_scopes(node):
        start  = node.start_pos
        parent = node.parent
        type   = parent.type

        is_param = (type == 'tfpdef' and parent.children[0] == node) or \
                   (type == 'param'  and parent.name == node)

        flow  = None
        scope = node

        while scope := scope.parent:
            t = scope.type

            if t in any_scope:
                if t in scopes:
                    if t in {"classdef", "funcdef", "lambdef"}:
                        if not is_param and scope.children[-2].start_pos >= start:
                            continue
                    elif t == "comp_for" and scope.children[1].type != "sync_comp_for":
                        continue
                    break

                elif not flow:
                    if t == "if_stmt":
                        children = scope.children
                        for index in compress(count(), map(is_keyword, children)):
                            if children[index].value != "else":
                                n = children[index + 1]
                                if start >= n.start_pos and start < n.end_pos:
                                    break
                        else:
                            flow = scope
                        continue

                    flow = scope

        node.scope_cache = {False: scope,
                            True:  flow or scope}
        # assert node.scope_cache[False] == get_parent_scope(node, include_flows=False)
        # assert node.scope_cache[True] == get_parent_scope(node, include_flows=True)
        return scope

    return map_parent_scopes


def optimize_get_parent_scope():
    from jedi.parser_utils import get_parent_scope
    from .lookup import map_parent_scopes
    from textension.utils import _copy_function

    get_parent_scope_org = _copy_function(get_parent_scope)
    
    def get_cached_parent_scope(node, include_flows=False):
        try:
            return node.scope_cache[include_flows]
        except AttributeError:
            return get_parent_scope_org(node, include_flows)
            # map_parent_scopes(node)
            # return node.scope_cache[include_flows]

    # _patch_function(get_parent_scope, get_cached_parent_scope)


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


# class Collection(list):
#     cached_data = {}

#     # XXX: What is this for? Do we need this?
#     def __getattr__(self, __name):
#         if __name == "value":
#             value = self[0].value
#             setattr(self, "value", value)
#             return value
#         raise AttributeError
   
#     # XXX: What is this for? Do we need this?
#     def get_cached(self, filter_func):
#         cached_data = self.cached_data
#         comp = tuple(self)
 
#         value = self.value
#         try:
#             prev_comp, dct = cached_data[value]
#         except:
#             dct = {}
#             cached_data[value] = comp, dct
#         else:
#             if prev_comp == comp:
#                 try:
#                     return dct[filter_func]
#                 except:
#                     pass
#         ret = dct[filter_func] = filter_func(self)
#         return ret


# TODO: Implement a get(string) method. Seems jedi requires this for in some cases.
# TAG:  UsedNames
class UsedNames(set):
    module: Module

    values = set.__iter__

    def get(self, name, default=()):
        try:
            return self.module.definitions[name]
        except:
            return default

    def __init__(self, module):
        self.module = module         # The module.

        self.module.updates = set()  # Updated nodes from the diff parser.
        self.nodes = set()           # ``module.children``.
        self.module.definitions = defaultdict(set)
        self.module.scope_cache = {False: None, True: None}

    def __getitem__(self, _):
        raise KeyError

    def update_module(self):
        from builtins import set, filter

        module   = self.module
        updates  = module.updates
        current  = set(module.children)

        is_basenode = BaseNode.__instancecheck__  # Means node has children.
        is_namenode = Name.__instancecheck__

        definitions = module.definitions

        for branch in current - self.nodes | updates:
            pool = [branch]

            # Process branch nodes.
            for node in filter(is_basenode, pool):
                map_parent_scopes(node)
                pool += node.children

            # Process Name nodes in branch.
            for name in filter(is_namenode, pool):
                map_parent_scopes(name)
                if name.is_definition():
                    definitions[name.value] |= {name}

        updates.clear()
        self.nodes = current
        return self


# XXX: Requires prepare_optimizations.
# XXX: Requires optimize_name_get_definition.
# Optimize Module.get_used_names.
def optimize_Module_get_used_names():

    # XXX: Unmanaged, but these are modules.
    lookup_cache = {}

    def get_used_names(self: Module):
        if (used_names := self._used_names) is not None:
            return used_names

        if self in lookup_cache:
            self._used_names = used_names = lookup_cache[self]
        else:
            self._used_names = used_names = lookup_cache[self] = UsedNames(self)
        return used_names.update_module()

    Module.get_used_names = get_used_names


# XXX: This optimization requires UsedNames, and UsedNames requires this!
# Optimizes _AbstractUsedNamesFilter.get, .values and ._convert_names.
# Inlines ``filter._get_definition_names``
# TAG: AbstractUsedNamesFilter
def optimize_AbstractUsedNamesFilter():
    from jedi.inference.filters import _AbstractUsedNamesFilter
    from jedi.inference.value.klass import ClassFilter
    from itertools import repeat, chain

    is_basenode = BaseNode.__instancecheck__  # Means node has children.
    is_namenode = Name.__instancecheck__

    from dev_utils import ncalls
    values_orig = ClassFilter.values

    def values(self: _AbstractUsedNamesFilter):

        # XXX: Only deal with ClassFilter for now.
        if isinstance(self, ClassFilter):
            definitions = self._parser_scope.get_root_node().definitions

            names = []
            pool = self._parser_scope.children[-1].children[:]
            for node in iter(pool):
                if is_basenode(node):
                    if node.type == "funcdef":
                        names += [node.children[1]]  # The method name.
                        continue
                    pool += node.children
                elif is_namenode(node):
                    if node in definitions[node.value]:
                        names += [node]

            ret = self._filter(names)
            return map(self.name_class, repeat(self.parent_context), ret)




            # pool = self._parser_scope.children[-1].children

            # for node in filter(is_basenode, pool):
            #     pool += node.children

            # names = []
            
            # for name in filter(is_namenode, pool):
            #     if name in definitions[name.value]:
            #         names += [name]

            # names = chain.from_iterable(definitions.values())


            ret = self._filter(names)
            return map(self.name_class, repeat(self.parent_context), ret)
        print("not a ClassFilter:", type(self))
        definitions = self._parser_scope.get_root_node().definitions
        names = chain.from_iterable(definitions.values())
        ret = self._filter(names)
        return map(self.name_class, repeat(self.parent_context), ret)
        return values_orig(self)
    
    _AbstractUsedNamesFilter.values = values

    def _convert_names(self: _AbstractUsedNamesFilter, names):
        return map(self.name_class, repeat(self.parent_context), names)

    _AbstractUsedNamesFilter._convert_names = _convert_names


def optimize_name_get_definition():
    from parso.python.tree import Name, _GET_DEFINITION_TYPES, _IMPORTS
    index = list.index

    def get_definition(self: Name, import_name_always=False, include_setitem=False):
        p: BaseNode = self.parent
        type = p.type

        if type in {"funcdef", "classdef", "except_clause"}:

            # ``funcdef`` or ``classdef``.
            children = p.children
            if self is children[1]:  # Same as ``self == parent.name``.
                return p

            # self is ``e`` part of ``except X as e``.
            elif type == "except_clause":
                if children[index(children, self) - 1].value == "as":
                    return p.parent  # The try_stmt.
            return None

        while p:
            if p.type in _GET_DEFINITION_TYPES:
                if self in p.get_defined_names(include_setitem) or \
                        import_name_always and p.type in _IMPORTS:
                    return p
                return None
            p = p.parent

    Name.get_definition  = get_definition
