
from jedi.inference.flow_analysis import REACHABLE, UNREACHABLE, get_flow_branch_keyword, _break_check

from textension.utils import factory, _patch_function


def apply():
    optimize_reachability_check()


@factory
def iter_flows(node, include_scopes=False):
    from parso.python.tree import Keyword
    from itertools import compress, count

    flows  = {"for_stmt", "if_stmt", "try_stmt", "while_stmt", "with_stmt"}
    scopes = {"classdef", "comp_for", "file_input", "funcdef",  "lambdef",
              "sync_comp_for"}

    any_scope = scopes | flows
    is_keyword = Keyword.__instancecheck__

    def iter_flows(node, include_scopes=False):
        if include_scopes and (parent := node.parent):
            type = parent.type
            is_param = (type == 'tfpdef' and parent.children[0] == node) or \
                       (type == 'param'  and parent.name == node)

        scope = node
        while scope := scope.parent:
            t = scope.type
            if t in any_scope:
                if t in scopes:
                    if not include_scopes:
                        break
                    if t in {"classdef", "funcdef", "lambdef"}:
                        if not is_param and scope.children[-2].start_pos >= node.start_pos:
                            continue
                    elif t == "comp_for" and scope.children[1].type != "sync_comp_for":
                        continue

                elif t == "if_stmt":
                    children = scope.children
                    for index in compress(count(), map(is_keyword, children)):
                        if children[index].value != "else":
                            n = children[index + 1]
                            start = node.start_pos
                            if start >= n.start_pos and start < n.end_pos:
                                break
                    else:
                        yield scope
                    continue
                yield scope

        # assert node.scope_cache[False] == get_parent_scope(node, include_flows=False)
        # assert node.scope_cache[True] == get_parent_scope(node, include_flows=True)
        return scope

    return iter_flows


class LazyCache(set):
    def contains(self, node):
        if node in self:
            return True

        add = self.add
        for n in self.gen:
            add(n)
            if n is node:
                return True
        return False

    def __init__(self, start_node):
        self.gen = iter_flows(start_node)


def optimize_reachability_check():
    from jedi.inference.flow_analysis import reachability_check
    from .flow_analysis import iter_flows
    
    def reachability_check_new(context, value_scope, node, origin_scope=None):
        first_flow_scope = node_flow_scopes = None

        if origin_scope is not None:
            branch_matches = True
            for flow_scope in iter_flows(origin_scope):
                # This loop is unlikely, so we don't make the lazy cache unless needed.
                if node_flow_scopes is None:
                    node_flow_scopes = LazyCache(node)

                if node_flow_scopes.contains(flow_scope):
                    node_keyword   = get_flow_branch_keyword(flow_scope, node)
                    origin_keyword = get_flow_branch_keyword(flow_scope, origin_scope)

                    if branch_matches := node_keyword == origin_keyword:
                        break

                    elif flow_scope.type == "if_stmt" or \
                        (flow_scope.type == "try_stmt" and origin_keyword == 'else' and node_keyword == 'except'):
                            return UNREACHABLE

            first_flow_scope = next(iter_flows(node, include_scopes=True), None)

            if branch_matches:
                while origin_scope:
                    if first_flow_scope is origin_scope:
                        return REACHABLE
                    origin_scope = origin_scope.parent

        if first_flow_scope is None:
            first_flow_scope = next(iter_flows(node, include_scopes=True), None)

        return _break_check(context, value_scope, first_flow_scope, node)

    _patch_function(reachability_check, reachability_check_new)
