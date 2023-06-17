

from jedi.inference.compiled.value import CompiledValueFilter, SignatureParamName
from jedi.inference.value.instance import ValueSet, NO_VALUES
from jedi.inference.signature import AbstractSignature
from jedi.inference.lazy_value import LazyKnownValues

from inspect import Parameter
from itertools import chain

from textension.utils import _context
from ..tools import runtime, state, make_instance_name, make_compiled_name, make_compiled_value, make_instance

from ..common import VirtualFilter, VirtualInstance, VirtualName, VirtualValue, get_mro_dict
from ._mathutils import float_vector_map

import bpy
import _bpy

import types


def apply():
    patch_AnonymousParamName_infer()


callable_types = (
    types.GetSetDescriptorType,
    types.MemberDescriptorType,
    types.BuiltinFunctionType,
    types.FunctionType,
    types.MethodDescriptorType
)

context_rna_pointer = _context.as_pointer()
_void = object()


# See doc/python_api/sphinx_doc_gen.py.
context_type_map = {
    # context_member: (RNA type, is_collection)
    "active_action": ("Action", False),
    "active_annotation_layer": ("GPencilLayer", False),
    "active_bone": ("EditBone", False),
    "active_file": ("FileSelectEntry", False),
    "active_gpencil_frame": ("GreasePencilLayer", True),
    "active_gpencil_layer": ("GPencilLayer", True),
    "active_node": ("Node", False),
    "active_object": ("Object", False),
    "active_operator": ("Operator", False),
    "active_pose_bone": ("PoseBone", False),
    "active_sequence_strip": ("Sequence", False),
    "active_editable_fcurve": ("FCurve", False),
    "active_nla_strip": ("NlaStrip", False),
    "active_nla_track": ("NlaTrack", False),
    "annotation_data": ("GreasePencil", False),
    "annotation_data_owner": ("ID", False),
    "armature": ("Armature", False),
    "asset_library_ref": ("AssetLibraryReference", False),
    "bone": ("Bone", False),
    "brush": ("Brush", False),
    "camera": ("Camera", False),
    "cloth": ("ClothModifier", False),
    "collection": ("LayerCollection", False),
    "collision": ("CollisionModifier", False),
    "curve": ("Curve", False),
    "dynamic_paint": ("DynamicPaintModifier", False),
    "edit_bone": ("EditBone", False),
    "edit_image": ("Image", False),
    "edit_mask": ("Mask", False),
    "edit_movieclip": ("MovieClip", False),
    "edit_object": ("Object", False),
    "edit_text": ("Text", False),
    "editable_bones": ("EditBone", True),
    "editable_gpencil_layers": ("GPencilLayer", True),
    "editable_gpencil_strokes": ("GPencilStroke", True),
    "editable_objects": ("Object", True),
    "editable_fcurves": ("FCurve", True),
    "fluid": ("FluidSimulationModifier", False),
    "gpencil": ("GreasePencil", False),
    "gpencil_data": ("GreasePencil", False),
    "gpencil_data_owner": ("ID", False),
    "curves": ("Hair Curves", False),
    "id": ("ID", False),
    "image_paint_object": ("Object", False),
    "lattice": ("Lattice", False),
    "light": ("Light", False),
    "lightprobe": ("LightProbe", False),
    "line_style": ("FreestyleLineStyle", False),
    "material": ("Material", False),
    "material_slot": ("MaterialSlot", False),
    "mesh": ("Mesh", False),
    "meta_ball": ("MetaBall", False),
    "object": ("Object", False),
    "objects_in_mode": ("Object", True),
    "objects_in_mode_unique_data": ("Object", True),
    "particle_edit_object": ("Object", False),
    "particle_settings": ("ParticleSettings", False),
    "particle_system": ("ParticleSystem", False),
    "particle_system_editable": ("ParticleSystem", False),
    "pointcloud": ("PointCloud", False),
    "pose_bone": ("PoseBone", False),
    "pose_object": ("Object", False),
    "scene": ("Scene", False),
    "sculpt_object": ("Object", False),
    "selectable_objects": ("Object", True),
    "selected_asset_files": ("FileSelectEntry", True),
    "selected_bones": ("EditBone", True),
    "selected_editable_actions": ("Action", True),
    "selected_editable_bones": ("EditBone", True),
    "selected_editable_fcurves": ("FCurve", True),
    "selected_editable_keyframes": ("Keyframe", True),
    "selected_editable_objects": ("Object", True),
    "selected_editable_sequences": ("Sequence", True),
    "selected_files": ("FileSelectEntry", True),
    "selected_ids": ("ID", True),
    "selected_nla_strips": ("NlaStrip", True),
    "selected_movieclip_tracks": ("MovieTrackingTrack", True),
    "selected_nodes": ("Node", True),
    "selected_objects": ("Object", True),
    "selected_pose_bones": ("PoseBone", True),
    "selected_pose_bones_from_active_object": ("PoseBone", True),
    "selected_sequences": ("Sequence", True),
    "selected_visible_actions": ("Action", True),
    "selected_visible_fcurves": ("FCurve", True),
    "sequences": ("Sequence", True),
    "soft_body": ("SoftBodyModifier", False),
    "speaker": ("Speaker", False),
    "texture": ("Texture", False),
    "texture_slot": ("TextureSlot", False),
    "texture_user": ("ID", False),
    "texture_user_property": ("Property", False),
    "ui_list": ("UIList", False),
    "vertex_paint_object": ("Object", False),
    "view_layer": ("ViewLayer", False),
    "visible_bones": ("EditBone", True),
    "visible_gpencil_layers": ("GPencilLayer", True),
    "visible_objects": ("Object", True),
    "visible_pose_bones": ("PoseBone", True),
    "visible_fcurves": ("FCurve", True),
    "weight_paint_object": ("Object", False),
    "volume": ("Volume", False),
    "world": ("World", False),
}


rna_py_type_map = {
    "STRING": str,
    "ENUM": str,
    "INT": int,
    "BOOLEAN": bool,
    "FLOAT": float
}

def get_context_instance(value_context):
    if not runtime.context_rna:
        runtime.context_rna = ContextInstance(bpy.types.Context.bl_rna, value_context)
    return runtime.context_rna


def get_rna_value(obj, value_context):
    if isinstance(obj, bpy.props._PropertyDeferred):
        return PropertyValue(getattr(bpy.types, obj.function.__name__), value_context)
    if not isinstance(obj, type) and obj.as_pointer() == context_rna_pointer:
        return get_context_instance(value_context)
    return RnaValue(obj, value_context)


def instance_from_rnadef(rnadef, context):
    name = rnadef.identifier
    if isinstance(rnadef, bpy.types.Function):
        return RnaFunction(rnadef, context).instance

    type = rnadef.type
    if type == 'POINTER':
        return RnaValue(rnadef.fixed_type, context).instance

    elif type == 'COLLECTION':
        return PropCollectionValue(rnadef, context).instance

    elif type in rna_py_type_map:
        # INT, FLOAT and BOOLEAN are only types that define ``array_length``.
        if type not in {'STRING', 'ENUM'} and rnadef.array_length != 0:
            if rnadef.subtype in float_vector_map:
                return MathutilsValue(float_vector_map[rnadef.subtype], context).instance
            return PropArrayValue(rna_py_type_map[type], context).instance

        return VirtualValue(rna_py_type_map[type], context).instance

    else:
        assert False, f"RnaFilter.values unhandled: {name} {type}"

def make_instance_name_from_rnadef(rnadef, context):
    return instance_from_rnadef(rnadef, context).as_name(rnadef.identifier)


get_dict = type.__dict__["__dict__"].__get__
get_mro = type.__dict__["__mro__"].__get__


def map_rna_instance_names(obj, context, is_instance):
    mapping = {}

    for name, value in get_mro_dict(obj).items():
        if isinstance(value, bpy.types.bpy_struct):
            mapping[name] = RnaValue(value.rna_type, context).instance.as_name(name)
        elif not is_instance or isinstance(value, callable_types):
            mapping[name] = make_compiled_name(value, context, name)
        else:
            mapping[name] = make_instance_name(value, context, name)
    return mapping


def map_rna(rna, context):
    mapping = {}
    for name, rnadef in chain(rna.functions.items(), rna.properties.items()):
        mapping[name] = instance_from_rnadef(rnadef, context).as_name(rnadef.identifier)
    return mapping


property_instance_map = {
    "StringProperty": str,
    "FloatProperty":  float,
    "IntProperty":    int,
    "BoolProperty":   bool,
    "EnumProperty":   str,
    "PointerProperty": None,
    "CollectionProperty": None,
    "FloatVectorProperty": None,  # Needs bpy_prop_array
    "IntVectorProperty": None,
    "BoolVectorProperty": None,
}


class PropertyFilter(VirtualFilter):
    def map_values(self, value, is_instance):
        if is_instance:
            name = value.obj.bl_rna.identifier
            obj = property_instance_map[name]
            assert obj
        return map_rna_instance_names(obj, value.as_context(), True)


class PropertyInstance(VirtualInstance):
    pass


class PropertyValue(VirtualValue):
    filter_cls   = PropertyFilter
    instance_cls = PropertyInstance


# This class implements the RNA value lookup.
class RnaFilter(VirtualFilter):
    def map_values(self, value, is_instance):
        rna = value.obj

        if rna != rna.bl_rna:  # Likely an RNA instance.
            rna = rna.bl_rna
            is_instance = True

        elif not is_instance:
            rna = bpy.types.Struct.bl_rna

        context = value.as_context()
        return map_rna(rna, context) | map_rna_instance_names(rna, context, is_instance)


# Represents an instance of an RnaValue.
class RnaInstance(VirtualInstance):
    @property
    def api_type(self):
        if isinstance(self.class_value.obj, bpy.types.Function):
            return "function"
        return super().api_type

    # For eg ``bpy.data.objects[0].copy().``.
    def py__call__(self, arguments):
        if self.api_type == "function":
            for rnadef in self.class_value.obj.parameters:
                if rnadef.is_output:
                    return ValueSet((make_instance_name_from_rnadef(rnadef, self.parent_context)._value,))
        return NO_VALUES

    # Implements subscript for bpy_prop_collections.
    def py__simple_getitem__(self, index):
        if isinstance(self.class_value, PropCollectionValue):
            instance = RnaValue(self.class_value.obj.fixed_type, self.parent_context).instance
            return ValueSet((instance,))
        return NO_VALUES

    def get_signatures(self):
        if isinstance(self.class_value, RnaFunction):
            return self.class_value.get_signatures()
        return ()


# RNA value created from struct RNA definition.
class RnaValue(VirtualValue):
    filter_cls   = RnaFilter
    instance_cls = RnaInstance

    @property
    def api_type(self):
        if isinstance(self.obj, bpy.types.Function):
            return "function"
        return "unknown"

    def get_signatures(self):
        if self.api_type == "function":
            return [RnaFunctionSignature(self, is_bound=False)]
        return ()

    def py__doc__(self):
        if not (doc := self.rna.description):
            print(f"Missing ``description`` for {self.rna}")
        return doc

    def py__name__(self):
        return self.obj.bl_rna.identifier


class RnaName(VirtualName):
    value_type = RnaValue


class RnaFunctionFilter(VirtualFilter):
    def map_values(self, value, is_instance):
        return map_rna_instance_names(bpy.types.bpy_func, value.as_context(), is_instance)


class RnaFunction(RnaValue):
    filter_cls = RnaFunctionFilter

    def get_signatures(self):
        return (RnaFunctionSignature(self, self, is_bound=True),)

    def get_param_names(self):
        ret = []
        rna = self.obj
        if rna.use_self:
            p = RnaFunctionParamName(value=self, name="self")
            ret += [p]

        for param in self.obj.parameters:
            if param.is_output:
                continue

            p = RnaFunctionParamName(value=self, param_rna=param,)
            ret += [p]
        return ret


class RnaFunctionParamName(SignatureParamName):
    def __init__(self, value, param_rna=None, name=""):
        self._param = param_rna
        self._value = value
        self._string_name = name
        self.kind_name = "POSITIONAL_ONLY"

        if param_rna and not param_rna.is_required:
            self.kind_name = "KEYWORD_ONLY"

    @property
    def string_name(self):
        if self._param is not None:
            return self._param.identifier
        return self._string_name

    @property
    def annotated(self):
        return getattr(self._param, "fixed_type", _void)

    @property
    def default(self):
        if self._param.is_required:
            return ""
        return f'={getattr(self._param, "default", "")}'

    def get_kind(self):
        return getattr(Parameter, self.kind_name)

    def infer(self):
        if (annotated := self.annotated) is not _void:
            return ValueSet((make_compiled_value(annotated, self.get_root_context()),))
        return NO_VALUES

    def to_string(self):
        s = self.string_name
        if self.annotated is not _void:
            import inspect
            fmt = inspect.formatannotation(type(self.annotated))
            s += ": " + fmt.replace("bpy_types", "bpy.types")
        return s + self.default

    def get_root_context(self):
        return self._value.parent_context.get_root_context()


class RnaFunctionSignature(AbstractSignature):
    def __init__(self, value, function_value=None, is_bound=False):
        self.value = value
        self.is_bound = is_bound
        self._function_value = function_value or value

    def bind(self, value):
        return RnaFunctionSignature(value, self._function_value, is_bound=True)


class MathutilsValue(VirtualValue):
    filter_cls = CompiledValueFilter


class PropArrayFilter(VirtualFilter):
    def map_values(self, value: "PropArrayValue", is_instance):
        return map_rna_instance_names(value.obj, value.as_context(), is_instance)


# bpy.types.Object.[bound_box | rotation_axis_angle | etc.]
class PropArrayValue(VirtualValue):
    filter_cls = PropArrayFilter

    def __init__(self, getitem_type, context):
        super().__init__(bpy.types.bpy_prop_array, context)
        self.getitem_type = getitem_type

    # Implements subscript.
    def py__simple_getitem__(self, index):
        return ValueSet((make_instance(self.getitem_type, self.parent_context),))

    # Implements for loop inference.
    def py__iter__(self, contextualized_node=None):
        return ValueSet((LazyKnownValues(self.py__simple_getitem__(None)),))


class PropCollectionFilter(VirtualFilter):
    def map_values(self, value, is_instance):
        obj     = value.obj
        context = value.as_context()
        mapping = map_rna_instance_names(bpy.types.bpy_prop_collection, context, is_instance)
        mapping |= map_rna_instance_names(obj, context, is_instance)

        if is_instance and obj.srna:
            mapping |= map_rna(obj.srna, context)
        return mapping


class PropCollectionValue(RnaValue):
    filter_cls = PropCollectionFilter


class ContextInstance(RnaInstance):
    def __init__(self, obj, context):
        self.class_value = RnaValue(obj, context)
        self._arguments  = None

        self.filter = ContextInstanceFilter((self.class_value, True))

    def get_filters(self, *_, **__):
        yield self.filter


def context_rna_name_from_string(string, context):
    name_str, is_collection = context_type_map.get(string, ('', False))
    if cls := getattr(bpy.types, name_str, None):
        name = RnaName.from_object(cls.bl_rna, context, string, instance=True)
        if is_collection:
            name = make_instance_name(list[name], context, string)
        return name
    return None


class ContextInstanceFilter(RnaFilter):
    def map_values(self, value, is_instance):
        mapping  = super().map_values(value, is_instance)
        context  = self.compiled_value.parent_context

        for name in _bpy.context_members()["screen"]:
            if rna_name := context_rna_name_from_string(name, context):
                mapping[name] = rna_name
        return mapping

    def get(self, string: str):
        try:
            return (self.mapping[string],)
        except KeyError:
            # These aren't part of the screen context.
            if string in context_type_map:
                context = self.compiled_value.parent_context
                return (context_rna_name_from_string(string, context),)
        return ()


# Patch jedi's anonymous parameter inference so that bpy.types.Operator
# method parameters can be automatically inferred.
def patch_AnonymousParamName_infer():
    from jedi.inference.names import AnonymousParamName, NO_VALUES, ValueSet
    from jedi.inference.value.instance import BoundMethod

    from ..tools import _get_unbound_super_method

    is_RnaValue = RnaValue.__instancecheck__
    infer_orig = AnonymousParamName.infer

    def infer_rna_param(rna_obj, meth_name, query_index):
        param_index = 0
        if rna_func := rna_obj.bl_rna.functions.get(meth_name):
            for param in rna_func.parameters:
                if param.is_output:
                    continue
                elif param_index == query_index:
                    return param
                param_index += 1
        return None
    
    def rna_value_from_tree_value(value, context):
        if value.is_compiled():
            return None

        for c in (c for b in value.py__bases__() for c in b.infer()):
            if not c.is_compiled():
                continue
            if not c.access_handle.access._obj is bpy.types.bpy_struct:
                continue
            cls_name = value.tree_node.children[1].value
            if cls := getattr(bpy.types, cls_name, None):
                return RnaValue(cls, context)
        return None


    def infer(self: AnonymousParamName):
        if ret := infer_orig(self):
            return ret

        func = self.function_value

        if not isinstance(func, BoundMethod):
            return NO_VALUES

        meth_name = func._wrapped_value.tree_node.children[1].value
        param = self.tree_name.parent

        # An RNA function doesn't include self in its parameters list.
        param_index = param.position_index - int(bool(func.is_bound_method()))
        context = self.parent_context

        for value in (v for b in func.instance.class_value.py__bases__()
                        for v in b.infer()):

            # Support inferring Operator from ``bpy_types.py``. Since this is
            # a tree node, we have to convert it into an RnaValue.
            if rna := rna_value_from_tree_value(value, context):
                value = rna

            # Only RnaValue parameters are inferred for now.
            if not is_RnaValue(value):
                continue

            if param := infer_rna_param(value.obj, meth_name, param_index):
                if param.identifier == "context":
                    instance = get_context_instance(self.parent_context)
                else:
                    instance = instance_from_rnadef(param, self.parent_context)
                return ValueSet((instance,))

        return NO_VALUES
    AnonymousParamName.infer = infer
