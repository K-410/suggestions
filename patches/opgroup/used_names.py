# This implements used names caching and partial updates.
# Reduces times for 4k LOC from 100ms to 1.5ms.

from collections import defaultdict
from operator import attrgetter

from parso.python.tree import Module
from parso.python.diff import (_get_next_leaf_if_indentation,
                                _update_positions,
                                _PositionUpdatingFinished,
                                _NodesTreeNode)

from textension.utils import make_default_cache
from ..tools import is_basenode, is_namenode, dict_items, starchain

get_direct_node = attrgetter("tree_node.parent")


def apply():
    optimize_Module_get_used_names()


def optimize_Module_get_used_names():
    Module.get_used_names = get_used_names
    _NodesTreeNode.finish = finish


def node_cache_fallback(self: dict, module):
    return self.setdefault(module, NamesCache(module))


NodeCache  = make_default_cache(node_cache_fallback)
node_cache = NodeCache()


class NamesCache(dict):
    def __init__(self, module):
        self.module  = module  # The module tree node, "file_input".
        self.nodes   = set()   # The last stored set of module base nodes.
        self.updates = set()   # Nodes tagged for update by the diff parser.

        self.values  = self.values()  # For faster dict.values() access.

    def update(self):
        # Get the current module base nodes.
        nodes = set(filter(is_basenode, self.module.children))

        # Remove unreferenced nodes.
        for node in self.nodes - nodes:
            del self[node]

        # Map the names. Only new and updated nodes are mapped.
        for node in nodes - self.nodes | self.updates:
            names = defaultdict(list)
            pool  = node.children[:]
            for n in filter(is_basenode, pool):
                pool += n.children
            for n in filter(is_namenode, pool):
                names[n.value] += [n]
            self[node] = dict_items(names)

        # Store the new set of module base nodes.
        self.nodes = nodes

        # Compose ``used_names`` from the module base nodes.
        result = defaultdict(list)
        for name, names in starchain(self.values):
            result[name] += names

        result.default_factory = None
        return result


def get_used_names(self: Module):
    if self._used_names is None:
        self._used_names = node_cache[self].update()
    return self._used_names


# Eliminates recursion and adds support for passing updated nodes to cache.
def finish(self: _NodesTreeNode):
    pool = [self]
    for node in iter(pool):
        pool += node._node_children

        tree_node = node.tree_node
        tree_node.children = children = []
        
        for prefix, group, line_offset, end_leaf in node._children_groups:
            first_leaf = _get_next_leaf_if_indentation(group[0].get_first_leaf())
            first_leaf.prefix = prefix + first_leaf.prefix
            if line_offset != 0:
                try:
                    _update_positions(group, line_offset, end_leaf)
                except _PositionUpdatingFinished:
                    pass
            children += group

        # Reset the parents
        for child in children:
            child.parent = tree_node

    # Add updated module base nodes to the names cache.
    node_cache[self.tree_node].updates = set(
        filter(is_basenode, map(get_direct_node, self._node_children)))
