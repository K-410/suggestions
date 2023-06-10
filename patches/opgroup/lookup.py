# This adds optimizations for used names lookup and filtering.

from collections import defaultdict
from textension.utils import _patch_function, factory, close_cells
from operator import attrgetter


def apply():
    # The order of these is important because of inheritance, yay!

    # optimize_ParserTreeFilter_filter()  # XXX: Broken. itertools imports Any (doesn't exist)
    optimize_ParserTreeFilter_values()
    # optimize_ClassFilter_values()
    optimize_GlobalNameFilter()
    # optimize_AbstractUsedNamesFilter()


# XXX: Required because we filter differently now.
def optimize_GlobalNameFilter():
    from jedi.inference.filters import GlobalNameFilter

    # from itertools import compress, chain
    # get_parent_type = attrgetter("parent.type")
    # is_global_statement = "global_stmt".__eq__

    def get(self: GlobalNameFilter, name):
        try:
            names = self._used_names[name]
        except KeyError:
            return []
        return self._convert_names(self._filter(names))

    def _filter(self: GlobalNameFilter, names):
        ret = []
        for name in names:
            if name.parent.type == 'global_stmt':
                ret += [name]
        return ret

    # XXX Broken
    # def values(self: GlobalNameFilter):
    #     parent_types = map(get_parent_type, chain.from_iterable(self._used_names.values()))  # ``name.parent.type``
    #     selectors    = map(is_global_statement, parent_types)  # ... == "global_stmt"
    #     global_names = compress(self._used_names, selectors)
    #     # TODO: Inline self._convert_names and make functional?
    #     return self._convert_names(global_names)

    GlobalNameFilter.get = get
    # GlobalNameFilter.values = values
    GlobalNameFilter._filter = _filter


# Optimizes ParserTreeFilter._filter.
# XXX: This is not a drop-in replacement. It only works in conjunction with lookup optimizations.
# Names are cached on the filter.
# The filter is cached on the node.
def optimize_ParserTreeFilter_filter():
    from jedi.inference.filters import flow_analysis, ParserTreeFilter
    from jedi.inference.flow_analysis import reachability_check
    from itertools import compress, takewhile, repeat
    from builtins import map, sorted

    get_start = attrgetter("start_pos")

    not_unreachable = flow_analysis.UNREACHABLE.__ne__
    not_reachable   = flow_analysis.REACHABLE.__ne__

    from .lookup import get_parent_scope_fast

    def _filter(self: ParserTreeFilter, names):
        # This inlines AbstractFilter._filter.
        if end := self._until_position:
            names = compress(names, map(end.__gt__, map(get_start, names)))

        # The scope which ``names`` is tested for reachability.
        scope = self._parser_scope

        tmp = []
        # Inlines ParserTreeFilter._is_name_reachable
        for name in names:
            parent = name.parent
            parent_type = parent.type

            if parent_type in {"classdef", "funcdef"}:
                parent_scope = get_parent_scope_fast(name.parent)
                if parent_scope is scope:
                    tmp += [name]

            else:
                parent_scope = get_parent_scope_fast(name)
                if parent_scope is scope:
                    tmp += [name]
                    continue


            # if parent_type != "trailer":
            #     if parent_type in {"classdef", "funcdef"}:
            #         parent_scope = get_parent_scope_fast(name.parent)
            #     if parent_scope is scope:
            #         tmp += [name]

        if tmp:
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


# XXX: This optimization requires UsedNames, and UsedNames requires this!
# Optimizes _AbstractUsedNamesFilter.get, .values and ._convert_names.
# Inlines ``filter._get_definition_names``
# TAG: AbstractUsedNamesFilter
def optimize_AbstractUsedNamesFilter():
    from jedi.inference.filters import _AbstractUsedNamesFilter
    from itertools import repeat

    values_orig = _AbstractUsedNamesFilter.values
    from dev_utils import measure_total

    def values(self: _AbstractUsedNamesFilter):
        print("Filter not optimized:", type(self))
        # with measure_total:
        ret = values_orig(self)
        return ret
        # definitions = self._parser_scope.get_root_node().definitions
        # names = chain.from_iterable(definitions.values())
        # ret = self._filter(names)
        # return map(self.name_class, repeat(self.parent_context), ret)
    
    _AbstractUsedNamesFilter.values = values

    def _convert_names(self: _AbstractUsedNamesFilter, names):
        return list(map(self.name_class, repeat(self.parent_context), names))

    _AbstractUsedNamesFilter._convert_names = _convert_names


def optimize_ParserTreeFilter_values():
    from jedi.inference.filters import ParserTreeFilter
    from itertools import repeat
    from ..tools import is_basenode, is_namenode

    values_orig = ParserTreeFilter.values

    stub_cache = {}

    def values(self: ParserTreeFilter):
        if self.parent_context.is_stub():
            module = self.parent_context.get_root_context().tree_node
            try:
                return stub_cache[module]
            except KeyError:
                return stub_cache.setdefault(module, list(values_orig(self)))

        names = []
        pool = [self._parser_scope]

        for n in filter(is_basenode, pool):
            pool += [n.children[1]] if n.type in {"classdef", "funcdef"} else n.children

        for n in filter(is_namenode, pool):
            if n.get_definition(include_setitem=True):
                names += [n]

        ret = list(self._filter(names))
        ret = list(map(self.name_class, repeat(self.parent_context), ret))
        return ret

    # XXX: This breaks ``[0].``.
    # ParserTreeFilter.values = values


def optimize_ClassFilter_values():
    from jedi.inference.value.klass import ClassFilter
    from jedi.inference.filters import ParserTreeFilter
    from itertools import repeat
    from ..tools import is_basenode, is_namenode

    from dev_utils import measure_total, ncalls
    values_orig = ParserTreeFilter.values

    def get_values(filter_):
        # XXX: Don't use filter_._parser_scope. It could be completely unrelated.
        class_node = filter_._class_value.tree_node
        pool = class_node.children[-1].children[:]

        names = []

        for n in filter(is_basenode, pool):
            pool += [n.children[1]] if n.type in {"classdef", "funcdef"} else n.children

        for n in filter(is_namenode, pool):
            parent = n.parent
            parent_type = parent.type
            if parent_type in {"decorator", "argument"}:
                continue

            # XXX: Without this ``type`` class gets typing members.
            if n.get_definition(include_setitem=True):
                names += [n]

        ret = filter_._filter(names)
        return map(filter_.name_class, repeat(filter_.parent_context), ret)

    stub_value_cache = {}

    def values(self: ClassFilter):
        value = self._class_value
        if value.is_stub():
            try:
                return stub_value_cache[value]
            except:
                ret = stub_value_cache[value] = list(get_values(self))
                return ret

        return get_values(self)

    ClassFilter.values = values

    # XXX: Temporary. SelfAttributeFilter doesn't work with optimized ClassFilter.values().

    # XXX: This is wrong. Why am i setting this on SelfAttributeFilter?????
    # SelfAttributeFilter.values = values_orig
