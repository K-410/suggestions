# This adds optimizations for:
# 
# Retained lookup:
# Each root (Module) has a persistent lookup instance. The lookup handles
# detection/mapping, and deletion of new and old nodes respectively. This
# is done by using set's union, subtraction and difference methods.
#
# Incremental updates:
# When the DiffParser finalizes a tree node, it is added to a set of updates
# which the lookup instance then adds to the list of nodes that must be mapped.
#
# Scope mapping:
# Consists of a Scope/string key pair which is used to fetch the names of a
# given scope. This map is managed by the lookup instance.


def apply_optimization():
    
    optimize_Module_get_used_names()
    optimize_get_definition_names()


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


# TAG:  ModuleLookup
class ModuleLookup(defaultdict):
    module: Module
    scope_map: dict

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

        # if new_nodes - old_nodes:
        #     print("new nodes", new_nodes - old_nodes)
        # We only care about new nodes.
        for branch in new_nodes - old_nodes:

            pool = [branch]
            branch_names = []

            # Process branch nodes.
            for node in filter(is_basenode, pool):
                pool += node.children

            # Process Name nodes in branch.
            for name in filter(is_namenode, pool):
                # if name.parent.type != "trailer":
                name.definition = {(False, False): _get_definition(name, False, False)}
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

    def __getitem__(self, string):
        # print("ModuleLookup.__getitem__", string, get_callsite_string(depth=3))
        # raise Exception
        # ncalls(sys._getframe(1).f_code)
        return super().__getitem__(string)


# Optimize Module.get_used_names.
def optimize_Module_get_used_names():
    assert is_optimized(prepare_optimizations)

    # XXX: Unmanaged, but these are modules.
    lookup_cache = {}

    # TAG: get_used_names
    @override_method(Module)
    def get_used_names(self: Module):
        if (used_names := self._used_names) is not None:
            return used_names
        if self in lookup_cache:
            self._used_names = used_names = lookup_cache[self]
        else:
            self._used_names = used_names = lookup_cache[self] = ModuleLookup(self)
        return used_names.update_module()


# Optimizes _get_definition_names in 'jedi/inference/filters.py'
def optimize_get_definition_names():
    assert is_optimized(optimize_name_get_definition)
    from parso.python.tree import Name

    # Same as Name.get_definition, but without trying the cache.
    get_definition = Name.get_definition

    # TAG: get_definition_names
    # The result is cached on the ``Collection`` object instead.

    def _get_definition_names_o(parso_cache_node, used_names: ModuleLookup, string):
        # Jedi sometimes looks up strings like ``__init__`` on classes that 
        # don't define them. In this case we just jedi sort it automagically.
        try:
            names = used_names.scope_map[parso_cache_node.node, string]
        except:
            print("can't find", repr(string), "on", parso_cache_node.node, get_callsite_string(depth=3))  # Debug purposes.
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

    filters._get_definition_names = _get_definition_names_o
