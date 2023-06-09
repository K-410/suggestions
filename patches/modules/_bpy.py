# This module implements import resolution for bpy.

from jedi.inference.names import SubModuleName
import bpy
import sys
import _bpy

from types import ModuleType
from itertools import repeat
from operator import getitem

from ._mathutils import float_vector_map
from ._bpy_types import MathutilsValue, PropArrayValue, RnaValue
from ..common import VirtualValue, VirtualModule, Importer_redirects, CompiledModule_redirects


app_submodule_names = tuple(
    name for name in bpy.app.__match_args__ if f"bpy.app.{name}" in sys.modules)


def apply():
    fix_bpy_imports()


prop_names = (
    "BoolProperty",
    "BoolVectorProperty",
    "CollectionProperty",
    "EnumProperty",
    "FloatProperty",
    "FloatVectorProperty",
    "IntProperty",
    "IntVectorProperty",
    "PointerProperty",
    "StringProperty",
)

# Map of PropertyFunctions for custom property inference.
prop_func_map: dict[str, "PropertyFunction"] = {}

prop_type_map = {
    "BoolProperty":       bool,
    "EnumProperty":       str,
    "FloatProperty":      float,
    "IntProperty":        int,
    "StringProperty":     str,
}

# ``bpy_prop_collection_idprop`` isn't reachable anywhere else.
for cls in bpy.types.bpy_prop_collection.__subclasses__():
    if cls.__name__ == "bpy_prop_collection_idprop":
        prop_type_map["CollectionProperty"] = cls
        break


def _as_module(obj, name):
    m = ModuleType(name)
    try:
        m.__dict__.update(obj.__dict__)
    except AttributeError:
        for name in dir(obj):
            m.__dict__[name] = getattr(obj, name)
    return m


def infer_vector_from_arguments(self: "PropertyFunction", arguments, context):
    for key, value in arguments.unpack():
        if key == "subtype" and value.data.type == "string":
            if obj := float_vector_map.get(value.data._get_payload()):
                return (MathutilsValue(obj, context).instance,)

    obj = prop_type_map[self.obj.__name__.replace("Vector", "")]
    return (PropArrayValue(obj, context).instance,)


# Needs to exist for import redirects.
class BpyModule(VirtualModule):
    pass


class AppModule(VirtualModule):
    def get_submodule_names(self, only_modules=False):
        yield from map(SubModuleName, repeat(self.as_context()), app_submodule_names)


class OpsModule(VirtualModule):
    def get_submodule_names(self, only_modules=False):
        sub = map(getitem, map(str.partition, _bpy.ops.dir(), repeat("_OT_")), repeat(0))
        names = set(map(str.lower, set(sub)))
        return list(map(SubModuleName, repeat(self.as_context()), names))


class PropsModule(VirtualModule):
    def py__getattribute__(self, name_or_str, **kw):
        if value := prop_func_map.get(name_or_str.value):
            return (value,)
        return super().py__getattribute__(name_or_str)


class VirtualFunction(VirtualValue):
    def py__call__(self, arguments):
        instance = VirtualValue(bpy.props._PropertyDeferred, self.as_context()).instance
        def instance_call(arguments):
            return (VirtualValue(str, self.as_context()).instance,)
        instance.py__call__ = instance_call
        return (instance,)



class PropertyFunction(VirtualFunction):
    def py__call__(self, arguments):
        instance = VirtualValue(bpy.props._PropertyDeferred, self.as_context()).instance
        instance.py__call__ = lambda *_, **__: self.py_instance__call__(arguments)
        return (instance,)

    def py_instance__call__(self, arguments):
        context = self.as_context()
        func_name = self.obj.__name__

        if obj := prop_type_map.get(func_name):
            return (VirtualValue(obj, context).instance,)

        elif "Vector" in func_name:
            return infer_vector_from_arguments(self, arguments, context)

        obj = getattr(bpy.types, func_name).bl_rna
        return (RnaValue(obj, context).instance,)


def fix_bpy_imports():
    Importer_redirects["bpy"]     = BpyModule(bpy)
    Importer_redirects["bpy.ops"] = OpsModule(bpy.ops)

    # Add bpy.app redirect.
    Importer_redirects["bpy.app"] = AppModule(_as_module(bpy.app, "bpy.app"))

    # Add bpy.app handlers | icons | timers | translations.
    for name in app_submodule_names:
        full_name = f"bpy.app.{name}"
        submodule = _as_module(getattr(bpy.app, name), full_name)
        Importer_redirects[full_name] = VirtualModule(submodule)

    # Add bpy.props redirect.
    Importer_redirects["bpy.props"] = PropsModule(bpy.props)
    Importer_redirects["_bpy.props"] = Importer_redirects["bpy.props"]
    context = Importer_redirects["bpy.props"].as_context()

    # Add property function redirects.
    for name in prop_names:
        prop_func_map[name] = PropertyFunction(getattr(bpy.props, name), context)

    # Add redirects for compiled value interceptions.
    for module in Importer_redirects.values():
        CompiledModule_redirects[module.obj] = module
