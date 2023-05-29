# This module implements import resolution for bpy.

from jedi.inference.names import SubModuleName
import bpy
import sys
import _bpy

from types import ModuleType
from .common import VirtualValue, VirtualModule, Importer_redirects, CompiledModule_redirects
from itertools import repeat
from operator import getitem
from .modules._mathutils import float_vector_map
from .modules._bpy_types import MathutilsValue, PropArrayValue, RnaValue

# This doesn't seem to exist anywhere.
bpy_prop_collection_idprop = bpy.types.bpy_prop_collection.__subclasses__()[0]


# Make ``bpy.app`` into a virtual module.
app_module = VirtualModule(ModuleType("bpy.app"))

for name in bpy.app.__match_args__:
    full_name = f"bpy.app.{name}"
    value = getattr(bpy.app, name)

    # For handlers, timers, translations etc.
    if full_name in sys.modules and not isinstance(value, ModuleType):
        value = ModuleType(full_name)
        value.__dict__.update({sub_k: getattr(value, sub_k) for sub_k in dir(value)})
    app_module.obj.__dict__[name] = value


# Make ``bpy.ops`` into a virtual module.
class OpsModule(VirtualModule):
    def get_submodule_names(self, only_modules=False):
        sub = map(getitem, map(str.partition, _bpy.ops.dir(), repeat("_OT_")), repeat(0))
        names = set(map(str.lower, set(sub)))
        return list(map(SubModuleName, repeat(self.as_context()), names))


# Make ``bpy.props`` into a virtual module.
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


def infer_vector_from_arguments(self: "PropertyFunction", arguments, context):
    for key, value in arguments.unpack():
        if key == "subtype" and value.data.type == "string":
            if obj := float_vector_map.get(value.data._get_payload()):
                return (MathutilsValue(obj, context).instance,)

    obj = prop_type_map[self.obj.__name__.replace("Vector", "")]
    return (PropArrayValue(obj, context).instance,)


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


prop_type_map = {
    "BoolProperty":       bool,
    "CollectionProperty": bpy_prop_collection_idprop,
    "EnumProperty":       str,
    "FloatProperty":      float,
    "IntProperty":        int,
    "StringProperty":     str,
}


props_module = PropsModule(bpy.props)
ops_module   = OpsModule(bpy.ops)
props_module = PropsModule(bpy.props)
# bpy_private_module = VirtualModule(_bpy)

Importer_redirects["bpy.app"] = app_module
Importer_redirects["bpy.ops"] = ops_module
# Importer_redirects["_bpy.props"] = props_module
Importer_redirects["bpy.props"] = props_module
# Importer_redirects["_bpy"] = bpy_private_module

for module in Importer_redirects.values():
    CompiledModule_redirects[module.obj] = module

props_context = props_module.as_context()


prop_func_map = {
    "BoolProperty":        PropertyFunction(bpy.props.BoolProperty, props_context),
    "BoolVectorProperty":  PropertyFunction(bpy.props.BoolVectorProperty, props_context),
    "CollectionProperty":  PropertyFunction(bpy.props.CollectionProperty, props_context),
    "EnumProperty":        PropertyFunction(bpy.props.EnumProperty, props_context),
    "FloatProperty":       PropertyFunction(bpy.props.FloatProperty, props_context),
    "FloatVectorProperty": PropertyFunction(bpy.props.FloatVectorProperty, props_context),
    "IntProperty":         PropertyFunction(bpy.props.IntProperty, props_context),
    "IntVectorProperty":   PropertyFunction(bpy.props.IntVectorProperty, props_context),
    "PointerProperty":     PropertyFunction(bpy.props.PointerProperty, props_context),
    "StringProperty":      PropertyFunction(bpy.props.StringProperty, props_context),
}
