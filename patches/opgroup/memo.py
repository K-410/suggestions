# This adds optimizations to jedi's caching.

from ..common import state_cache, state_cache_kw, state_cache_default


def apply():
    optimize_memos()


is_func = type(apply).__instancecheck__


# This just unwraps wrappers, grabbing the underlying function.
def unwrap(func):
    assert func.__closure__

    funcs = []
    for cell in func.__closure__:
        contents = cell.cell_contents
        if is_func(contents):
            funcs += contents,

    assert len(funcs) == 1, f"Ambiguous unwrap, got {len(funcs)}: {funcs}"
    return funcs[0]


def optimize_memos():
    from ..common import NO_VALUES
    from jedi.inference.base_value import LazyValueWrapper, safe_property
    @safe_property
    @state_cache
    def _wrapped_value(self):
        return self._get_wrapped_value()
    LazyValueWrapper._wrapped_value = _wrapped_value

    from jedi.inference.imports import Importer
    from jedi.inference.names import ImportName

    @state_cache
    def infer(self):
        m = self._from_module_context
        return Importer(m.inference_state, [self.string_name], m, level=self._level).follow()

    ImportName.infer = infer

    from jedi.inference.gradual.base import DefineGenericBaseClass
    DefineGenericBaseClass.get_generics = state_cache(unwrap(DefineGenericBaseClass.get_generics))

    from jedi.inference.base_value import _ValueWrapperBase
    _ValueWrapperBase.create_cached = classmethod(
        state_cache_kw(unwrap(_ValueWrapperBase.create_cached.__func__))
    )

    from jedi.inference.imports import infer_import
    from textension.utils import _patch_function, _copy_function

    # Requires default value recursion mitigation.
    func = infer_import.__closure__[1].cell_contents
    _patch_function(infer_import, state_cache_default(NO_VALUES)(func))

    from jedi.inference.param import get_executed_param_names_and_issues
    f = _copy_function(get_executed_param_names_and_issues)
    _patch_function(get_executed_param_names_and_issues, state_cache(f))

    from jedi.inference.names import TreeNameDefinition
    TreeNameDefinition.infer = state_cache(TreeNameDefinition.infer)