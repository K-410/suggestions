# This makes optimizations to ClassValue types.

from jedi.inference.value.klass import ClassMixin, ClassValue


get_filters_orig = ClassValue.get_filters


def apply():
    pass
    optimize_ClassMixin_get_filters()


class ClassFilterCache(dict):
    def __missing__(self, key):
        instance, kw = key
        result = list(get_filters_orig(instance, **dict(kw)))
        self[key] = result
        return result


def optimize_ClassMixin_get_filters():

    cache = ClassFilterCache()

    def get_filters(self: ClassValue, **kw):
        if self.parent_context.is_stub():
            return cache[self, tuple(kw.items())]

        return get_filters_orig(self, **kw)

    ClassMixin.get_filters = get_filters