# This module implements patches required for jedi to work with bpy.


from .tools import _get_super_meth, _deferred_patch_function, _copy_func


def apply():
    patch_importer_completion_names()
    patch_get_api_type()
    patch_getattr_paths()
    patch_anonymous_param()
    patch_get_builtin_module_names()
    patch_module_value_getattribute_fallback()
    # patch_builtin_from_name()
    # patch_compiledvaluefilter()
    patch_compiledvalue()
    patch_compiledmodule()
    patch_classmixin()
    # patch_misc()
    patch_stubfilter()
    # patch_try_to_load_stub()


# Patch Importer.completion_names to resolve builtin submodules and
# submodules with a dot in their name. Blender does this for modules like
# bpy.app, bpy.app.handlers, mathutils.noise, mathutils.geometry etc.
# Fixes 'import bpy.', 'from bpy.' completions.
def patch_importer_completion_names():
    from jedi.inference.compiled.value import CompiledValue, CompiledModule
    from jedi.inference.gradual.conversion import convert_values
    from jedi.inference.names import ImportName, SubModuleName
    from jedi.inference.imports import Importer
    from sys import modules as sys_modules
    from os.path import isdir, join

    # This just moves the flask ext-related stuff out for sake of clarity.
    def flask_ext_names(self):
        for mod in self._get_module_names():
            modname = mod.string_name
            if modname.startswith('flask_'):
                yield ImportName(self._module_context, modname[len('flask_'):])
        for dir in self._sys_path_with_modifications(is_completion=True):
            flaskext = join(dir, 'flaskext')
            if isdir(flaskext):
                yield from self._get_module_names([flaskext])

    def completion_names(self, inference_state, only_modules=False):
        if not self._infer_possible:
            return []

        if not self.import_path:
            path = self._fixed_sys_path if self.level else None
            return self._get_module_names(search_path=path)

        names = []
        if self._str_import_path == ('flask', 'ext'):
            names.extend(flask_ext_names(self))

        values = self.follow()
        for value in values:
            api_type = value.api_type
            if api_type in {"module", "namespace", "bpy_app"}:
                if api_type == "module":
                    names.extend(value.sub_modules_dict().values())

                elif api_type == "bpy_app":
                    names.extend([SubModuleName(value.as_context(), n)
                        for n in value.access_handle.access._obj.__match_args__
                        if f"bpy.app.{n}" in sys_modules])

        if not only_modules:
            names.extend(e for c in values | convert_values(values)
                           for f in c.get_filters() for e in f.values())
        return names

    Importer.completion_names = completion_names


# Patches 'DirectObjectAccess.get_api_type' to identify 'bpy.app' as a namespace.
# Required for 'patch_importer_completion_names' to work. Improves performance.
def patch_get_api_type():
    from ...types import is_module, is_builtin, is_method, is_class, is_function, is_methoddescriptor
    from jedi.inference.compiled.access import DirectObjectAccess
    import bpy

    is_bpyapp = type(bpy.app).__instancecheck__

    def get_api_type(self):
        obj = self._obj
        if is_class(obj):
            return 'class'
        elif is_module(obj):
            return 'module'
        elif is_builtin(obj) or is_method(obj) or is_methoddescriptor(obj) or is_function(obj):
            return 'function'
        elif is_bpyapp(obj):
            return 'bpy_app'
        return 'instance'

    DirectObjectAccess.get_api_type = get_api_type


# Patch getattr_paths so any rna type is intercepted and resolved using RnaResolver.
def patch_getattr_paths():
    import builtins
    from inspect import getmodule, ismodule
    from warnings import catch_warnings, simplefilter

    from jedi.inference.compiled.access import DirectObjectAccess, _sentinel
    from .resolvers import RnaResolver
    from bpy.types import bpy_struct

    def getattr_paths(self, name, default=_sentinel):
        # This part is identical to the start of the original function
        try:
            with catch_warnings(record=True):
                simplefilter("always")
                return_obj = getattr(self._obj, name)
        except Exception as e:
            if default is _sentinel:
                if isinstance(e, AttributeError):
                    raise
                raise AttributeError
            return_obj = default

        # This part redirects rna instance resolutions
        if isinstance(return_obj, bpy_struct):
            return_obj = RnaResolver(return_obj)

        # This part is identical to the rest of the original function
        access = self._create_access(return_obj)
        if ismodule(return_obj):
            return [access]
        try:
            module = return_obj.__module__
        except AttributeError:
            pass
        else:
            if module is not None and isinstance(module, str):
                try:
                    __import__(module)
                except ImportError:
                    pass
        module = getmodule(return_obj)
        if module is None:
            module = getmodule(type(return_obj))
            if module is None:
                module = builtins
        return [self._create_access(module), access]

    DirectObjectAccess.getattr_paths = getattr_paths


def patch_anonymous_param():
    """Patch jedi's anonymous parameter inference so that bpy.types.Operator
    method parameters can be automatically inferred.
    """
    from jedi.inference.base_value import NO_VALUES, ValueSet
    from jedi.inference.compiled import CompiledValue
    from jedi.inference.compiled.access import create_access
    from jedi.inference.names import AnonymousParamName
    from jedi.inference.value.instance import BoundMethod
    from ...types import is_operator
    from .resolvers import RnaResolver
    import bpy.types as bpy_types

    def get_super_method(cls: object, methname: str):
        for base in cls.__mro__[1:]:
            try:
                return object.__getattribute__(base, methname)
            except AttributeError:
                continue
        raise AttributeError(f"Method '{methname}' not found")

    meth_names = {"invoke", "execute", "modal", "draw", "poll"}
    super_infer = get_super_method(AnonymousParamName, "infer")
    # TODO: Could be extended to other base classes eg. Panel, Menu etc.
    is_meth = BoundMethod.__instancecheck__

    def infer(self: AnonymousParamName):
        ret = super_infer(self)
        if ret:
            return ret

        val = self.function_value
        if not is_meth(val) or val.name.string_name not in meth_names:
            return NO_VALUES

        param = self.tree_name.search_ancestor("param")
        for base in val.instance.class_value.py__bases__():
            for cls in base.infer():
                try:
                    assert is_operator(cls.access_handle.access._obj.bl_rna)
                except:
                    continue
                if param.position_index == 1:
                    obj = bpy_types.Context.bl_rna
                else:
                    obj = bpy_types.Event.bl_rna
                access = create_access(val.inference_state,
                                       RnaResolver(obj, static=True))
                c = CompiledValue(val.inference_state, access, None)
                return ValueSet({c})
        return NO_VALUES
    AnonymousParamName.infer = infer


# Patch jedi's builtin modules getter to include file-less modules that
# aren't officially builtin.. Fixes "import _bpy", among others.
def patch_get_builtin_module_names():
    from jedi.inference.compiled.subprocess import functions
    import sys
    args = sys.modules.items(), sys.builtin_module_names

    def get_builtin_module_names(_, args=args):
        items, builtins = args
        modules = tuple(n for n, m in items if not hasattr(m, "__file__"))
        return set(builtins + modules)

    functions.get_builtin_module_names.__code__ = get_builtin_module_names.__code__
    functions.get_builtin_module_names.__defaults__ = get_builtin_module_names.__defaults__


# Jedi fails at py__getattribute__ some non-stub modules like 'types', but
# provides a means to use fallbacks. This implements that.
# This fixes 'from types import FunctionType' when prefer_stubs is False.
def patch_module_value_getattribute_fallback():
    from jedi.inference.value.module import ModuleValue
    from jedi.inference.base_value import ValueSet
    from jedi.inference.compiled.value import CompiledValue
    from jedi.inference.compiled.access import create_access
    from sys import modules as sys_modules

    def py__getattribute__alternatives(self, name_or_str):
        try:
            obj = getattr(sys_modules[self.string_names[0]], name_or_str)
            access = create_access(self.inference_state, obj)
            return ValueSet((CompiledValue(self.inference_state, access, None),))
        except:
            # Mostly for debug.
            print(f"failed py__getattribute__alternatives for {name_or_str} ({self})")
        return ValueSet([])

    ModuleValue.py__getattribute__alternatives = py__getattribute__alternatives


def patch_misc():
    import importlib
    import os
    import bpy

    org_path_stat = importlib._bootstrap_external._path_stat

    def restore():
        importlib._bootstrap_external._path_stat = org_path_stat
        cache.clear()

    def init(path):
        importlib._bootstrap_external._path_stat = fast_path_stat
        bpy.app.timers.register(restore)
        return fast_path_stat(path)

    # 357    0.378    0.001    4.424    0.012
    class Cache(dict):
        def __missing__(self, path):
            ret = self[path] = stat(path)
            return ret

    stat = os.stat
    cache = Cache()
    fast_path_stat = cache.__getitem__

    importlib._bootstrap_external._path_stat = init


# Required for non-stub modules
def patch_builtin_from_name():
    from jedi.inference import compiled
    def builtin_from_name(inference_state, string):
        filter_ = next(inference_state.builtins_module.get_filters())
        name, = filter_.get(string)
        value, = name.infer()
        return value
    compiled.builtin_from_name.__code__ = builtin_from_name.__code__



# def patch_compiledvaluefilter():
#     from jedi.inference.compiled import CompiledValueFilter, CompiledName
#     def get(self, name, *args, check_has_attribute=False):
#         if args:
#             allowed_getattr_callback, in_dir_callback = args
#             has_attribute, is_descriptor = allowed_getattr_callback(name, safe=False)
#             if check_has_attribute and not has_attribute or self.is_instance and not in_dir_callback(name):
#                 return []
#         return [CompiledName(self._inference_state, self.compiled_value, name)]
#     CompiledValueFilter.get = CompiledValueFilter._get = get



# Patches CompiledValue to support inference and extended py__call.
def patch_compiledvalue():
    from jedi.inference.compiled.value import CompiledValue, create_from_name, NO_VALUES
    from jedi.inference.compiled import builtin_from_name
    from jedi.inference.value import CompiledInstance
    from jedi.inference.base_value import ValueSet

    org_py__call__ = CompiledValue.py__call__
    compiled_py__call__overrides = {
        list.copy: "list",
        set.copy: "set",
        dict.copy: "dict",
    }

    def infer(self):
        return ValueSet({self})

    def infer_compiled_value(self):
        return create_from_name(self._inference_state, self._parent_value, self.string_name)

    def py__call__(self, arguments):
        if ret := org_py__call__(self, arguments):
            return ret
        elif self.parent_context._value.access_handle.access._obj.__name__ == "builtins":
            if obj := compiled_py__call__overrides.get(self.access_handle.access._obj):
                obj = builtin_from_name(self.inference_state, obj)
                return ValueSet((CompiledInstance(self.inference_state, self.parent_context, obj, arguments),))
        return NO_VALUES

    CompiledValue.infer = infer
    CompiledValue.infer_compiled_value = infer_compiled_value
    CompiledValue.py__call__ = py__call__


# Patch CompiledModule to support getting sub-modules.
def patch_compiledmodule():
    from jedi.inference.compiled.value import CompiledModule
    from jedi.inference.names import SubModuleName
    from importlib.util import find_spec
    from types import ModuleType
    from sys import modules
    is_module = ModuleType.__instancecheck__
    is_str = str.__instancecheck__
    true_dir = object.__dir__
    sentinel = object()

    module_name_cache = {}

    def find_name(mod):
        name = None
        try:
            name = mod.__name__
        except:
            for sys_modname, sys_mod in modules.items():
                if mod is sys_mod:
                    name = sys_modname
                    break
        # It's possible that 'name' isn't even a string.
        if not is_str(name):
            name = str(type(mod).__name__)
        return name

    def sub_modules_dict(self):
        mod = self.access_handle.access._obj
        names = {}

        try:
            name = module_name_cache[mod]
        except:
            name = module_name_cache[mod] = find_name(mod)

        if is_module(mod):
            all_exports = set(true_dir(mod))
            # __all__ could hold anything, and an overridden dir may even
            # fail. Only object.__dir__ promises to return a list of strings.
            try: all_exports |= {str(v) for v in mod.__all__}
            except: pass
            try: all_exports |= {str(v) for v in dir(mod)}
            except: pass

            # We skip __init__ because it has potential side effects.
            for m in all_exports - {"__init__"}:
                n = f"{name}.{m}"
                try:
                    assert find_spec(n)

                except (ValueError, AssertionError, ModuleNotFoundError):
                    if n not in modules or getattr(mod, m, sentinel) is not modules[n]:
                        continue
                names[m] = SubModuleName(self.as_context(), m)
        return names

    CompiledModule.sub_modules_dict = sub_modules_dict


# Patches ClassMixin to support compiled value filters.
def patch_classmixin():
    from jedi.inference.value.klass import ClassMixin, ClassFilter
    
    def get_filters(self, origin_scope=None, is_instance=False,
                    include_metaclasses=True, include_type_when_class=True):
        if include_metaclasses:
            if metaclasses := self.get_metaclasses():
                yield from self.get_metaclass_filters(metaclasses, is_instance)
        is_wrapped = hasattr(self, "access_handle")
        from jedi.inference.compiled.value import CompiledValueFilter
        for cls in self.py__mro__():
            if cls.is_compiled():
                if is_wrapped:
                    yield CompiledValueFilter(self.inference_state, self, is_instance)
                else:
                    yield from cls.get_filters(is_instance=is_instance)
            else:
                yield ClassFilter(self, node_context=cls.as_context(), origin_scope=origin_scope, is_instance=is_instance)
    ClassMixin.get_filters = get_filters


# Patch StubFilter to lookup names using its compiled counterpart as fallback.
# Specifically this fixes completing sys._getframe, among other stub values.
def patch_stubfilter():
    from jedi.inference.gradual.stub_value import StubFilter
    super_meth = _get_super_meth(StubFilter, "_is_name_reachable")

    private_cache = {}

    def _is_name_reachable(self: StubFilter, name):
        if not super_meth(self, name):
            return False

        definition = name.get_definition()
        if definition.type in {'import_from', 'import_name'}:
            if name.parent.type not in {'import_as_name', 'dotted_as_name'}:
                return False

        n = name.value
        # TODO: Possible optimization here
        if n.startswith('_') and not (n.startswith('__') and n.endswith('__')):

            # Jedi blanket rejects private names from stubs, which means
            # omitting things like sys._getframe. This fixes that.
            try:
                cache = private_cache[self]
            except:
                merged_dict = {}
                for nsv in self.parent_context._value.non_stub_value_set:
                    merged_dict |= nsv.access_handle.access._obj.__dict__
                cache = private_cache[self] = merged_dict
            return n in cache
        return True

    StubFilter._is_name_reachable = _is_name_reachable


# Patch _try_to_load_stubs to find stubs based the actual suffix of a module.
# Finds stubs for modules which embed their cpython version in their name.
def patch_try_to_load_stub():
    from importlib.machinery import all_suffixes
    from jedi.inference.gradual.typeshed import \
        _try_to_load_stub, _try_to_load_stub_from_file, FileIO
    from os import access, F_OK

    suffixes = tuple(all_suffixes())
    stub_loader = _copy_func(_try_to_load_stub)

    def run_first(inference_state, import_names, python_value_set,
                  parent_module_value, sys_path):
        for c in python_value_set:
            try:
                file_path = c.py__file__()
            except:
                continue
            if not file_path:
                continue
            name = file_path.name
            endswith = name.endswith
            if not endswith(suffixes):
                continue

            suf = max([suf for suf in suffixes if endswith(suf)], key=len)
            path = str(file_path.parent.joinpath(name[:-len(suf)]  + ".pyi"))
            if access(path, F_OK):
                m = _try_to_load_stub_from_file(
                    inference_state, python_value_set,
                    file_io=FileIO(path), import_names=import_names)
                if m is not None:
                    return m
        return stub_loader(
            inference_state, import_names, python_value_set, parent_module_value, sys_path)

    _deferred_patch_function(_try_to_load_stub, run_first)
