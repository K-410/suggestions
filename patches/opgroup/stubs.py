from jedi.inference.gradual.stub_value import StubModuleValue, StubModuleContext, StubFilter
from jedi.inference.value.module import DictFilter

from textension.utils import make_default_cache


class CachedStubFilter(StubFilter):
    def __init__(self, parent_context):
        super().__init__(parent_context=parent_context)
        self.cache = {}

    def get(self, name):
        try:
            return self.cache[name]
        except KeyError:
            return self.cache.setdefault(name, list(super().get(name)))


def on_missing_stub_filter(self: dict, module: StubModuleValue):
    context = StubModuleContext(module)

    self[module] = filters = [
        CachedStubFilter(context),
        *module.iter_star_filters(),
        DictFilter(module.sub_modules_dict()),
        DictFilter(module._module_attributes_dict())
    ]
    return filters


stub_filter_cache = make_default_cache(on_missing_stub_filter)()


def apply():
    optimize_StubModules()


def optimize_StubModules():

    def get_filters(self: StubModuleValue, origin_scope=None):
        yield from stub_filter_cache[self]

    StubModuleValue.get_filters = get_filters

    StubModuleContext.is_stub = True.__bool__