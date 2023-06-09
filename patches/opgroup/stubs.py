from jedi.inference.gradual.stub_value import StubModuleValue, StubModuleContext, StubFilter
from jedi.inference.value.module import ModuleMixin, DictFilter


class CachedStubFilter(StubFilter):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.cache = {}

    def get(self, name):
        try:
            return self.cache[name]
        except KeyError:
            return self.cache.setdefault(name, list(super().get(name)))


class FilterCache(dict):
    def __missing__(self, key):
        module, origin_scope = key

        main = CachedStubFilter(parent_context=module.as_context(),
                                origin_scope=origin_scope)

        self[key] = filters = [
            main,
           *module.iter_star_filters(),
            DictFilter(module.sub_modules_dict()),
            DictFilter(module._module_attributes_dict())
        ]
        return filters


# A cache of persistent stub filters.
cache = FilterCache()


def apply():
    optimize_StubModules()


def optimize_StubModules():

    def get_filters(self: StubModuleValue, origin_scope=None):
        yield from cache[self, origin_scope]

    StubModuleValue.get_filters = get_filters

    StubModuleContext.is_stub = True.__bool__