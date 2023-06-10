# This module implements optimizations for various filter types.

def apply():
    optimize_SelfAttributeFilter_values()
    optimize_ClassFilter_values()


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


def optimize_ClassFilter_values():
    from jedi.inference.flow_analysis import reachability_check, REACHABLE, UNREACHABLE
    from jedi.inference.value.klass import ClassFilter
    from itertools import repeat
    from ..tools import is_basenode, is_namenode, is_funcdef, is_classdef

    cache = {}

    def get_class_name_definitions(self: ClassFilter, scope, context):
        names = []
        pool  = scope.children[-1].children[:]

        for n in filter(is_basenode, pool):
            pool += [n.children[1]] if n.type in {"classdef", "funcdef"} else n.children

        # Get name definitions.
        for n in filter(is_namenode, pool):
            if n.get_definition(include_setitem=True):
                names += [n]

        names = self._filter(names)
        names = list(map(self.name_class, repeat(context), names))
        return names

    def values(self: ClassFilter):
        scope = self._parser_scope
        names = []

        if is_classdef(scope):
            context = self.parent_context

            if context.is_stub():
                if scope not in cache:
                    cache[scope] = get_class_name_definitions(self, scope, context)
                return cache[scope]
            else:
                pass

            names = get_class_name_definitions(self, scope, context)
        else:
            pass  # Should we do anything about non-class scopes?
        return names
    
    # This version differs from the stock ``_check_flows`` by:
    # - Doesn't sort entries. Not applicable in class filter contexts.
    # - Doesn't break after first hit. The stock version is was designed to
    #   process like-named sequences. Here we process *all* names in one go.
    def _check_flows(self, names):
        context = self._node_context
        value_scope = self._parser_scope
        origin_scope = self._origin_scope
        for name in names:
            check = reachability_check(context=context, value_scope=value_scope, node=name, origin_scope=origin_scope)
            if check is not UNREACHABLE:
                yield name

    ClassFilter.values = values
    ClassFilter._check_flows = _check_flows