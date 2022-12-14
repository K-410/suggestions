# This module implements utilities for Suggestions



def _deferred_patch_function(orig_fn, new_fn):
    # Evil, but the only reliable way.
    def func(*args, **kw):
        from sys import _getframe
        f = _getframe(0)
        new = f.f_code.co_consts[-1]
        assert (f_back := f.f_back)
        f_back.f_globals[f.f_code.co_name] = new
        return new(*args, **kw)

    orig_fn.__code__ = func.__code__.replace(
        co_name=orig_fn.__code__.co_name,
        co_consts=func.__code__.co_consts + (new_fn,))


def _get_super_meth(cls, methname: str):
    for base in cls.mro()[1:]:
        try:
            return getattr(base, methname)
        except AttributeError:
            continue
    raise Exception(f"Method '{methname}' not on any superclass")


def _copy_func(f):
    import types, functools
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
                           argdefs=f.__defaults__,
                           closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g
