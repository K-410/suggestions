# This module implements optimizations for used names by caching and applying
# partial updates to module tree nodes. Optimized filter methods that depend
# on this optimization are also added here.

from collections import defaultdict
from operator import attrgetter

from parso.python.tree import Module
from parso.python.diff import (_get_next_leaf_if_indentation,
                                _update_positions,
                                _PositionUpdatingFinished,
                                _NodesTreeNode)

from textension.utils import instanced_default_cache, dict_items, starchain
from ..tools import is_basenode, is_namenode, is_leaf


get_direct_node = attrgetter("tree_node.parent")


def apply():
    optimize_Module_get_used_names()
    optimize_GlobalNameFilter_values()


def optimize_Module_get_used_names():
    Module.get_used_names = get_used_names
    _NodesTreeNode.finish = finish


def optimize_GlobalNameFilter_values():
    from jedi.inference.filters import GlobalNameFilter
    from itertools import repeat, compress
    from builtins import filter, map
    from ..tools import is_basenode

    get_position = attrgetter("line", "column")

    def is_valid(stmt, branch):
        pool = branch.children[:]
        for n in filter(is_basenode, pool):
            if stmt in filter(is_basenode, n.children):
                return True
            pool += filter(is_basenode, n.children)
        return False

    def get_direct_node(node):
        while (node := node.parent).parent:
            pass
        return node

    def values(self: GlobalNameFilter):
        nodes = node_cache[self._parser_scope]
        globs = nodes.globals
        names = []

        for glob in reversed(globs):
            node = get_direct_node(glob)

            if node in nodes and is_valid(glob, node):
                names += glob.children[1::2]
            else:
                globs.remove(glob)

        if pos := self._until_position:
            names = compress(names, map(pos.__gt__, map(get_position, names)))
        return map(self.name_class, repeat(self.parent_context), names)

    GlobalNameFilter.values = values


def node_cache_fallback(self: dict, module):
    return self.setdefault(module, NamesCache(module))


node_cache = instanced_default_cache(node_cache_fallback)


class NamesCache(dict):
    def __init__(self, module):
        self.module  = module  # The module tree node, "file_input".
        self.nodes   = set()   # The last stored set of module base nodes.
        self.updates = set()   # Nodes tagged for update by the diff parser.

        self.values  = self.values()  # For faster dict.values() access.
        self.globals = []

    def update(self):
        # Get the current module base nodes.
        nodes = set(filter(is_basenode, self.module.children))

        # Remove unreferenced nodes.
        for node in self.nodes - nodes:
            del self[node]

        globals_ = self.globals

        # Map the names. Only new and updated nodes are mapped.
        for node in nodes - self.nodes | self.updates:
            names = defaultdict(list)
            pool  = node.children[:]
            for n in filter(is_basenode, pool):
                pool += n.children
                if n.type == "global_stmt":
                    globals_ += n,

            for n in filter(is_namenode, pool):
                names[n.value] += n,
            self[node] = dict_items(names)

        # Store the new set of module base nodes.
        self.nodes = nodes

        # Compose ``used_names`` from the module base nodes.
        self.result = result = defaultdict(list)
        for name, names in starchain(self.values):
            result[name] += names

        result.default_factory = None
        return result


def get_used_names(self: Module):
    used_names = node_cache[self]

    if self._used_names is None:
        self._used_names = False
        used_names.update()
    return used_names.result


# Eliminates recursion and adds support for passing updated nodes to cache.
def finish(self: _NodesTreeNode):
    pool = [self]
    for node in iter(pool):
        pool += node._node_children
        children = []
        
        for prefix, group, line_offset, end_leaf in node._children_groups:
            first_leaf = _get_next_leaf_if_indentation(group[0].get_first_leaf())
            first_leaf.prefix = prefix + first_leaf.prefix
            if line_offset != 0:
                try:
                    _update_positions(group, line_offset, end_leaf)
                except _PositionUpdatingFinished:
                    pass
            children += group

        tree_node = node.tree_node
        tree_node.children = children
        for child in children:
            child.parent = tree_node

    # Add updated module base nodes to the names cache.
    node_cache[self.tree_node].updates = set(
        filter(is_basenode, map(get_direct_node, self._node_children)))
