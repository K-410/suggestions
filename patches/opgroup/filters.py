# This module implements optimizations for various filter types.
from jedi.inference.flow_analysis import reachability_check, UNREACHABLE
from jedi.inference.value.klass import ClassFilter
from itertools import repeat
from operator import attrgetter

from ..tools import is_basenode, is_namenode


def apply():
    optimize_SelfAttributeFilter_values()
    optimize_ClassFilter_values()
    optimize_AnonymousMethodExecutionFilter()
    optimize_ParserTreeFilter_values()


# Optimizes SelfAttributeFilter.values to search within the class scope.
def optimize_SelfAttributeFilter_values():
    from jedi.inference.value.instance import SelfAttributeFilter
    from ..tools import is_basenode, is_namenode, is_funcdef, is_classdef
    from itertools import repeat

    def values(self: SelfAttributeFilter):
        names   = []
        scope   = self._parser_scope

        # We only care about classes.
        if is_classdef(scope):
            # Stubs never have self definitions.
            context = self.parent_context

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


def get_scope_name_definitions(self: ClassFilter, scope, context):
    names = []
    pool  = scope.children[:]

    for n in filter(is_basenode, pool):
        pool += [n.children[1]] if n.type in {"classdef", "funcdef"} else n.children

    # Get name definitions.
    for n in filter(is_namenode, pool):
        if n.get_definition(include_setitem=True):
            names += [n]

    return list(self._filter(names))


# This version differs from the stock ``_check_flows`` by:
# - Doesn't break after first hit. The stock version is designed to process
#   like-named sequences. Here we process *all* names in one go.
# - No pre-sorting. It's not applicable unless we break out of the loop.
def _check_flows(self, names):
    context = self._node_context
    value_scope = self._parser_scope
    origin_scope = self._origin_scope
    for name in names:
        check = reachability_check(context=context, value_scope=value_scope, node=name, origin_scope=origin_scope)
        if check is not UNREACHABLE:
            yield name


def optimize_ClassFilter_values():
    from ..tools import is_classdef

    stub_classdef_cache = {}

    def values(self: ClassFilter):
        context = self.parent_context
        scope   = self._parser_scope
        names   = []

        if is_classdef(scope):
            # The class suite.
            scope = scope.children[-1]

            if context.is_stub():
                if scope not in stub_classdef_cache:
                    stub_classdef_cache[scope] = get_scope_name_definitions(self, scope, context)
                names = stub_classdef_cache[scope]
            else:
                names = get_scope_name_definitions(self, scope, context)
        return list(map(self.name_class, repeat(context), names))

    ClassFilter.values = values
    ClassFilter._check_flows = _check_flows


def optimize_AnonymousMethodExecutionFilter():
    from jedi.inference.value.instance import AnonymousMethodExecutionFilter
    from ..tools import is_funcdef

    cache = {}

    def get(self: AnonymousMethodExecutionFilter, name):
        # ret = get_orig(self, name)
        scope = self._parser_scope
        assert is_funcdef(scope)

        if scope not in cache:
            cache[scope] = get_scope_name_definitions(self, scope, self.parent_context)

        names = [n for n in cache[scope] if n.value == name]
        names = self._convert_names(names)
        return names

    AnonymousMethodExecutionFilter.get = get
    AnonymousMethodExecutionFilter._check_flows = _check_flows


def optimize_ParserTreeFilter_values():
    from jedi.inference.filters import ParserTreeFilter

    def values(self: ParserTreeFilter):
        scope   = self._parser_scope
        context = self.parent_context

        names = get_scope_name_definitions(self, scope, context)
        return self._convert_names(names)
    
    ParserTreeFilter.values = values
    ParserTreeFilter._check_flows = _check_flows


def optimize_ParserTreeFilter_filter():
    from jedi.inference.flow_analysis import reachability_check
    from jedi.inference.filters import flow_analysis, ParserTreeFilter
    from itertools import compress, takewhile, repeat
    from builtins import map

    get_start = attrgetter("start_pos")

    not_unreachable = flow_analysis.UNREACHABLE.__ne__
    not_reachable   = flow_analysis.REACHABLE.__ne__

    def _filter(self: ParserTreeFilter, names):
        if end := self._until_position:
            names = compress(names, map(end.__gt__, map(get_start, names)))

        scope = self._parser_scope

        for name in names:
            if name.parent.type in {"classdef", "funcdef"}:
                if get_parent_scope_fast(name.parent) is not scope:
                    print("not in scope (function)")
            else:
                if get_parent_scope_fast(name) is not scope:
                    print("not in scope")

        if names:
            to_check = map(reachability_check,
                           repeat(self._node_context),
                           repeat(self._parser_scope),
                           names,
                           repeat(self._origin_scope))
            selectors = takewhile(not_reachable, map(not_unreachable, to_check))
            return compress(names, selectors)
        return []

    ParserTreeFilter._filter = _filter


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