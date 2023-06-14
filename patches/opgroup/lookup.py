# This adds optimizations for used names lookup and filtering.


def apply():
    pass


def optimize_AbstractUsedNamesFilter():
    from jedi.inference.filters import _AbstractUsedNamesFilter
    from itertools import repeat

    def _convert_names(self: _AbstractUsedNamesFilter, names):
        return list(map(self.name_class, repeat(self.parent_context), names))

    _AbstractUsedNamesFilter._convert_names = _convert_names
