import sys
import bpy
from types import BuiltinMethodType
from bpy import types as bpy_types
from typing import List, Tuple, GenericAlias, _GenericAlias, Any, Callable
from textension import utils
_context = utils._context
bpy_struct: type = bpy_types.bpy_struct

# Return type lookups for compiled values
restype_lookup = {}

iter_lookup = {}

subscript_lookup = {}

attr_lookup = {}
_AliasTypes = (GenericAlias, _GenericAlias)

# Adds missing return types to bmesh api
def set_bmesh_lookups():

    import mathutils
    import bmesh
    from bmesh.types import BMesh, BMVert, BMEdge, BMFace, BMLoop, \
        BMDeformVert, BMVertSeq, BMEdgeSeq, BMFaceSeq, BMLayerAccessFace, BMLoopSeq

    # return_type_lookup[BMDeformVert.get]              = Optional[float]
    # return_type_lookup[BMDeformVert.items]            = List[Tuple[int, float]]
    # return_type_lookup[BMDeformVert.keys]             = List[int]
    # return_type_lookup[BMDeformVert.values]           = List[float]
    # return_type_lookup[BMEdge.calc_face_angle]        = float
    # return_type_lookup[BMEdge.calc_face_angle_signed] = float
    # return_type_lookup[BMEdge.calc_length]            = float
    # return_type_lookup[BMEdge.calc_tangent]           = mathutils.Vector
    # return_type_lookup[BMEdge.other_vert]             = Optional[BMVert]
    # return_type_lookup[BMEdgeSeq.get]                 = Optional[BMEdge]
    # return_type_lookup[BMEdgeSeq.new]                 = BMEdge

    # return_type_lookup[BMesh.calc_loop_triangles]     = List[Tuple[BMLoop]]
    restype_lookup[BMesh.calc_loop_triangles]     = list[tuple[BMLoop]]

    # return_type_lookup[BMesh.calc_volume]             = float
    # return_type_lookup[BMesh.copy]                    = BMesh
    restype_lookup[bmesh.new]                    = BMesh

    # Sequence subscripts
    # return_type_lookup[BMVertSeq.__getitem__]         = BMVert
    # return_type_lookup[BMEdgeSeq.__getitem__]         = BMEdge
    # return_type_lookup[BMFaceSeq.__getitem__]         = BMFace

    # # Sequence iterators must point to themselves.
    # return_type_lookup[BMVertSeq.__iter__]            = BMVertSeq
    # return_type_lookup[BMEdgeSeq.__iter__]            = BMEdgeSeq
    # return_type_lookup[BMFaceSeq.__iter__]            = BMFaceSeq


    # iter_lookup[BMLoop.link_loops] = BMLoop
    # iter_lookup[BMVert.link_edges] = BMEdge
    # iter_lookup[BMVert.link_faces] = BMFace
    # iter_lookup[BMVert.link_loops] = BMLoop
    # iter_lookup[BMEdge.link_faces] = BMFace
    # iter_lookup[BMEdge.link_loops] = BMLoop
    # iter_lookup[BMEdge.verts]      = BMVert
    # iter_lookup[BMFace.loops]      = BMLoop
    # iter_lookup[BMFace.verts]      = BMVert
    # iter_lookup[BMesh.verts]       = BMVert
    # iter_lookup[BMesh.edges]       = BMEdge
    # iter_lookup[BMesh.faces]       = BMFace
    # iter_lookup[BMesh.loops]       = BMLoop

    # The inferred type of the iterator data
    # iter_lookup[BMVertSeq]         = BMVert
    # iter_lookup[BMEdgeSeq]         = BMEdge
    # iter_lookup[BMFaceSeq]         = BMFace

    # Subscripts
    # subscript_lookup[BMesh.verts]       = BMVert
    # subscript_lookup[BMesh.faces]       = BMFace
    # subscript_lookup[BMesh.edges]       = BMEdge
    # subscript_lookup[BMVert.link_edges] = BMEdge
    # subscript_lookup[BMVert.link_faces] = BMFace
    # subscript_lookup[BMVert.link_loops] = BMLoop
    # subscript_lookup[BMEdge.link_faces] = BMFace
    # subscript_lookup[BMEdge.link_loops] = BMLoop
    # subscript_lookup[BMEdge.verts]      = BMVert
    # subscript_lookup[BMFace.loops]      = BMLoop
    # subscript_lookup[BMFace.verts]      = BMVert
    # subscript_lookup[BMFace.edges]      = BMEdge

    # Attributes (getset_descriptors)
    # attr_lookup[BMesh.verts] = BMVertSeq
    # attr_lookup[BMesh.faces] = BMFaceSeq
    # attr_lookup[BMesh.edges] = BMEdgeSeq

    # attr_lookup[BMVert.link_edges] = BMEdgeSeq
    # attr_lookup[BMVert.link_faces] = BMFaceSeq
    # attr_lookup[BMVert.link_loops] = BMLoopSeq

    # attr_lookup[BMEdge.link_faces] = BMFaceSeq
    # attr_lookup[BMEdge.link_loops] = BMLoopSeq

    # attr_lookup[BMVert.co]   = mathutils.Vector

    # # bm.faces
    # attr_lookup[BMFaceSeq.active]     = Optional[BMFace]
    # attr_lookup[BMFaceSeq.layers]     = BMLayerAccessFace

ANSI_YELLOW     = '\033[33m'
ANSI_RED        = '\033[31m'
ANSI_GREEN      = '\033[32m'
ANSI_DEFAULT    = '\033[0m'

DEBUG_NONE = 0
DEBUG_RESOLUTION = 1
DEBUG_INFO = 2


DEBUG_LEVEL = DEBUG_NONE


def dbg_info(*args):
    if DEBUG_LEVEL > 1:
        dbg(*args)

def dbg(*args):
    if DEBUG_LEVEL > 0:
        print(*args)


def red(*args: str) -> str:
    items = [str(s) for s in args]
    return f"{ANSI_RED}{' '.join(items)}{ANSI_DEFAULT}"

def yellow(*args: str) -> str:
    items = [str(s) for s in args]
    return f"{ANSI_YELLOW}{' '.join(items)}{ANSI_DEFAULT}"

def green(*args: str) -> str:
    items = [str(s) for s in args]
    return f"{ANSI_GREEN}{' '.join(items)}{ANSI_DEFAULT}"


# Global context members are fetched via bpy.types.Context.bl_rna.properties.
# Non-global members use dynamic lookup (ctx_data_pointer_get in context.c),
# which means they have to be added manually.

# Obviously this sucks, but the complete listing can be found here:
# https://docs.blender.org/api/current/bpy.context.html

# Avoid getattr(context, member) as the main resolution fallback, because it
# may be NoneType, which is less useful than getting no completion at all.

# Except for Screen Context, dynamic members will never show up in context,
# because doing so would add nearly 100 entries which may not even exist.
buttons_context = {
    "texture_slot":             bpy_types.TextureSlot,
    "scene":                    bpy_types.Scene,
    "world":                    bpy_types.World,
    "object":                   bpy_types.Object,
    "mesh":                     bpy_types.Mesh,
    "armature":                 bpy_types.Armature,
    "lattice":                  bpy_types.Lattice,
    "curve":                    bpy_types.Curve,
    "meta_ball":                bpy_types.MetaBall,
    "light":                    bpy_types.Light,
    "speaker":                  bpy_types.Speaker,
    "lightprobe":               bpy_types.LightProbe,
    "camera":                   bpy_types.Camera,
    "material":                 bpy_types.Material,
    "material_slot":            bpy_types.MaterialSlot,
    "texture":                  bpy_types.Texture,
    "texture_user":             bpy_types.ID,
    "texture_user_property":    bpy_types.Property,
    "bone":                     bpy_types.Bone,
    "edit_bone":                bpy_types.EditBone,
    "pose_bone":                bpy_types.PoseBone,
    "particle_system":          bpy_types.ParticleSystem,
    "particle_system_editable": bpy_types.ParticleSystem,
    "particle_settings":        bpy_types.ParticleSettings,
    "cloth":                    bpy_types.ClothModifier,
    "soft_body":                bpy_types.SoftBodyModifier,
    "fluid":                    None,  # Not implemented
    "collision":                bpy_types.CollisionModifier,
    "brush":                    bpy_types.Brush,
    "dynamic_paint":            bpy_types.DynamicPaintModifier,
    "line_style":               bpy_types.FreestyleLineStyle,
    "collection":               bpy_types.LayerCollection,
    "gpencil":                  bpy_types.GreasePencil,
    "curves":                   None,  # Not implemented
    "volume":                   bpy_types.Volume,
}

clip_context = {
    "edit_movieclip":   bpy_types.MovieClip,
    "edit_mask":        bpy_types.Mask
}

file_context = {
    "active_file":               bpy_types.FileSelectEntry,
    "selected_files":       list[bpy_types.FileSelectEntry],
    "asset_library_ref":         bpy_types.AssetLibraryReference,
    "selected_asset_files": list[bpy_types.FileSelectEntry],
    "id":                        bpy_types.ID,
}

image_context = {
    "edit_image":       bpy_types.Image,
    "edit_mask":        bpy_types.Mask,
}

node_context = {
    "selected_nodes": list[bpy_types.Node],
    "active_node":         bpy_types.Node,
    "light":               bpy_types.Light,
    "material":            bpy_types.Material,
    "world":               bpy_types.World,
}

screen_context = {
    "scene":                            bpy_types.Scene,
    "visible_objects":             list[bpy_types.Object],
    "selectable_objects":          list[bpy_types.Object],
    "selected_objects":            list[bpy_types.Object],
    "selected_editable_objects":   list[bpy_types.Object],
    "editable_objects":            list[bpy_types.Object],
    "objects_in_mode":             list[bpy_types.Object],
    "objects_in_mode_unique_data": list[bpy_types.Object],
    "visible_bones":               list[bpy_types.EditBone],
    "editable_bones":              list[bpy_types.EditBone],
    "selected_bones":              list[bpy_types.EditBone],
    "selected_editable_bones":     list[bpy_types.EditBone],
    "visible_pose_bones":          list[bpy_types.PoseBone],
    "selected_pose_bones":         list[bpy_types.PoseBone],
    "selected_pose_bones_from_active_object": list[bpy_types.PoseBone],
    "active_bone":                      bpy_types.EditBone,
    "active_pose_bone":                 bpy_types.PoseBone,
    "active_object":                    bpy_types.Object,
    "object":                           bpy_types.Object,
    "edit_object":                      bpy_types.Object,
    "sculpt_object":                    bpy_types.Object,
    "vertex_paint_object":              bpy_types.Object,
    "weight_paint_object":              bpy_types.Object,
    "image_paint_object":               bpy_types.Object,
    "particle_edit_object":             bpy_types.Object,
    "pose_object":                      bpy_types.Object,
    "active_sequence_strip":            bpy_types.Sequence,
    "sequences":                   list[bpy_types.Sequence],
    "selected_sequences":          list[bpy_types.Sequence],
    "selected_editable_sequences": list[bpy_types.Sequence],
    "active_nla_track":                 bpy_types.NlaTrack,
    "active_nla_strip":                 bpy_types.NlaStrip,
    "selected_nla_strips":         list[bpy_types.NlaStrip],
    "selected_movieclip_tracks":   list[bpy_types.MovieTrackingTrack],
    "gpencil_data":                     bpy_types.GreasePencil,
    "gpencil_data_owner":               bpy_types.ID,
    "annotation_data":                  bpy_types.GreasePencil,
    "annotation_data_owner":            bpy_types.ID,
    "active_gpencil_layer":        list[bpy_types.GPencilLayer],
    "active_annotation_layer":          bpy_types.GPencilLayer,
    "active_gpencil_frame":        list[bpy_types.GPencilLayer],
    "visible_gpencil_layers":      list[bpy_types.GPencilLayer],
    "editable_gpencil_layers":     list[bpy_types.GPencilLayer],
    "editable_gpencil_strokes":    list[bpy_types.GPencilStroke],
    "active_operator":                  bpy_types.Operator,
    "active_action":                    bpy_types.Action,
    "selected_visible_actions":    list[bpy_types.Action],
    "selected_editable_actions":   list[bpy_types.Action],
    "editable_fcurves":            list[bpy_types.FCurve],
    "visible_fcurves":             list[bpy_types.FCurve],
    "selected_editable_fcurves":   list[bpy_types.FCurve],
    "selected_visible_fcurves":    list[bpy_types.FCurve],
    "active_editable_fcurve":           bpy_types.FCurve,
    "selected_editable_keyframes": list[bpy_types.Keyframe],
    "asset_library_ref":                bpy_types.AssetLibraryReference,
    "ui_list":                          bpy_types.UIList,
}

sequencer_context = {
    "edit_mask": bpy_types.Mask,
}

text_context = {
    "edit_text": bpy_types.Text
}

view3d_context = {
    "active_object": bpy_types.Object,
    "selected_ids": list[bpy_types.ID],
}


dyn_context = (
    buttons_context   |
    clip_context      |
    file_context      |
    image_context     |
    node_context      |
    screen_context    |
    sequencer_context |
    text_context      |
    view3d_context
)


# bpy.types.Property instances that get coerced to python types.
rna_pytypes = {
    "STRING": "",     # StringProperty
    "ENUM":   "",     # EnumProperty
    "BOOLEAN": False, # BoolProperty
    "INT":     0,     # IntProperty
    "FLOAT": 0.0,     # FloatProperty
}


propcoll_keys = [k for k in dir(bpy_types.bpy_prop_collection)]
bpy_struct_keys = [k for k in dir(bpy_struct) if not k.startswith("__")]
resolver_py__dict__ = {}
stub_cache = {}
resolver_cache = {}

# Faster than super cell getattribute. Important for lookups.
type_getattr = type.__getattribute__

struct_id = bpy_types.ID.bl_rna.functions['copy'].parameters['id'].fixed_type
return_type_cache = {}


class ResolverMeta(type):
    """The only reason this exists is because jedi refuses to infer return
    types that are instances (see jedi.inference.compiled.getattr_static).
    So we give jedi what it wants, while also resolving lookups.
    """
    def __getattribute__(cls, name):
        # RnaResolver.store holds all listings for any instance rna.
        if name == "__dict__":
            return type_getattr(cls, "store")

        # Queries meant for the resolver for property/function lookups.
        try:
            return getattr(type_getattr(cls, "resolver"), name)
        except:
            pass
        # When all else fails, query on the class itself.
        return type_getattr(cls, name)


# Generate a stub from a bpy_func instance with annotated return type.
def stub_from_rna_func(rna, fn):
    try:
        return stub_cache[rna, fn]
    except:
        pass

    dbg_info(green("creating stub"))
    # TODO: support is_required, is_never_none, is_argument_optional
    param = {}

    for p in fn.parameters:
        if p.is_output:
            dbg_info("  found output", p)

            # Return type is a basic python type, eg. str/int
            if p.type in rna_pytypes:
                ret = type(rna_pytypes[p.type])

            # Return type is a Blender type
            else:
                ret_rna = p.fixed_type

                # The "copy" method has "bpy.types.ID" as its return type
                # regardless of the refined struct RNA type. This workaround
                # makes the return type the same as the parent rna.
                if fn.identifier == "copy" and ret_rna == struct_id:
                    ret_rna = rna

                # Jedi expects the return type to be a class so we need to
                # use metaclasses to provide return type resolutions.
                try:
                    ret = return_type_cache[ret_rna]
                except:
                    class ReturnType(metaclass=ResolverMeta):
                        name = f"Return Type Resolver {rna.identifier}.{fn.identifier}()"
                        resolver = RnaResolver(ret_rna, static=True)
                        __module__ = "builtins"
                        store = resolver.store
                    ret = return_type_cache[ret_rna] = ReturnType
            param["return"] = ret
        else:
            dbg_info("  found param", p)
            param[p.identifier] = None

    ret = param.pop("return", None)
    stub = stub_from(ret, fn.identifier, tuple(param), fn.description)
    stub_cache[rna, fn] = stub
    return stub


def resolve_rna(resolver, rna, attr) -> Any:  # NOTE: Leave this annotation be.
    if isinstance(rna, bpy_types.Property):         # vscode highlights bug workaround.
        rna_type = rna.type

        if rna_type == 'POINTER':  # Pointer to an RNA type
            dbg("is a pointer")
            ret = RnaResolver(rna.fixed_type, static=True)

        elif rna_type == 'COLLECTION':  # Pointer to a property collection
            dbg("is a Collection")
            dbg(f"make CollectionResolver for {green(rna.name)}")
            ret = CollectionResolver(rna, resolver)
        else:
            dbg("is a Pytype")
            ret = rna_pytypes[rna_type]

    elif isinstance(rna, bpy_types.Function):
        dbg("is a Function")
        ret = stub_from_rna_func(resolver, rna)

    elif isinstance(rna, (Callable, BuiltinMethodType)):
        dbg("is a Method")
        ret = rna

    # Dynamic additions like context resolver
    elif isinstance(rna, RnaResolver):
        dbg("is a Resolver")
        ret = rna

    else:
        if getattr(rna, "__objclass__", None) is bpy_struct:
            type(rna)
        # raise AttributeError(f"unhandled: {attr} {rna} {type(rna)}")
        dbg(yellow(f"unhandled: {attr}", rna, type(rna)), resolver)
        return rna

    dbg(f"resolved to {green(ret)}\n")
    return ret


def print_traceback():  # XXX: For development
    import traceback
    print(traceback.format_exc())


collection_resolver_cache = {}

collection_dict = {}
for cls in reversed(bpy.types.bpy_prop_collection.__mro__):
    collection_dict.update(cls.__dict__)


is_getset_descriptor = type(type.__dict__["__dict__"].__get__).__instancecheck__


class CollectionResolver:
    """Resolves collection properties. Some collections don't have a bl_rna
    component, so they can't be resolved using RnaResolver. CollectionResolver
    also enables subscription and iterator access.
    """

    # Required for subscript/iterator
    __class__ = list

    def __new__(cls, rna, data):
        # The rna type expected
        assert isinstance(rna, bpy.types.CollectionProperty)

        try:
            return collection_resolver_cache[rna]
        except:
            pass

        self = super().__new__(cls)
        collection_resolver_cache[rna] = self

        self.store = collection_dict.copy()

        # XXX: "id_data" is difficult to resolve without walking bpy_types,
        # visiting all rna types and map every collection. Blender just
        # doesn't store their refined type in bl_rna.
        self.data = data

        srna = rna.srna
        if srna is not None:
            # Resolver for this collection
            resolver = RnaResolver(srna, static=True)
            resolver.store.update(self.store)
            self.store.update(srna.properties.items() + srna.functions.items())
            self.store["bl_rna"] = rna
            self.resolver = resolver
        else:
            dbg_info(red("srna is None for", rna))
            # print(red("srna is None for", rna))
            # raise AssertionError(red("srna is None for", rna))

        # srna = rna.srna
        # if srna is None:
        #     srna = rna.fixed_type
        #     print(red("srna is None for", rna))
        # # Resolver for this collection
        # resolver = RnaResolver(srna, static=True)
        # resolver.store.update(store)
        # store.update(srna.properties.items() + srna.functions.items())
        # store["bl_rna"] = rna
        # self.resolver = resolver
        #     # raise AssertionError(red("srna is None for", rna))

        # Resolver for the element type in the collection
        self.restype = restype = resolver_as_class(RnaResolver(rna.fixed_type, static=True))

        # Required for 'for loops'
        self.__iter__ = stub_from(restype, "__iter__")
        return self

    # def __getattribute__(self, name):
    #     dbg_info("__getattribute__", name)
    #     return super().__getattribute__(name)

    def __getattr__(self, name):
        dbg_info("__getattr__", repr(name))

        # Static resolution for bpy_prop_collection methods
        if name == "get":
            return stub_from(self.restype)
        elif name == "items":
            return stub_from(List[Tuple[str, self.restype]])
        elif name == "values":
            return stub_from(List[self.restype])

        # Resolver wasn't defined. It means srna was None.
        elif name == "resolver":
            ret = getattr(self.restype, name)
            ret = resolve_rna(None, ret, name)
            return ret
            
        # return super().__getattribute__(name)
        # raise AttributeError(name=name)
        try:
            ret = getattr(self.resolver, name)
            # ret = self.store[name]
        except:
            dbg("failed getting", name)
            raise AttributeError
        else:
            try:
                ret = resolve_rna(None, ret, name)
                return ret
            except:
                ret = getattr(self.resolver, name)
                return ret
        #     if isinstance(ret, bpy_struct):
        #         ret = RnaResolver(ret, static=True)
        #         ret = getattr(ret, name)
        #         print("return bl_rna", ret)
        #         return ret

        #     if is_getset_descriptor(ret):
        #         print("is getset descriptor", name, ret)
        #         if name == "data":
        #             print("returning data")
        #             return self.data

    # Required for subscript
    def __iter__(self):
        return iter((self.restype,))

    def __dir__(self):
        return list(self.store.keys())

    def __repr__(self):
        return f"<CollectionResolver: {self.restype}>"


class RnaResolver:
    """RnaResolver provides resolutions for RNA instances by listing bl_rna
    methods and properties and resolving them as instance attributes in order
    to provide completions.
    """

    store: dict

    def __new__(cls, rna, *, static=False):
        if is_resolver(rna) or (not static and rna.as_pointer() == rna.bl_rna.as_pointer()):
            return rna

        try:
            return resolver_cache[rna.bl_rna]
        except KeyError:
            bl_rna = rna.bl_rna

            self = super().__new__(cls)
            resolver_cache[bl_rna] = self

            self.identifier = bl_rna.identifier

            store = bpy_struct.__dict__ | type(rna).__dict__
            store.update(bl_rna.properties.items() + bl_rna.functions.items())
            self.store = store
            return self

    def __getattribute__(self, name):
        self._frame = sys._getframe(0)
        return super().__getattribute__(name)

    def __getattr__(self, name):  # Resolve only if AttributeError is raised.
        dbg(f"{self} query {yellow(name)}")

        try:
            data = self.store[name]
        except KeyError:
            dbg(self, red(name, "not in store:"))
            # print(self, red(name, "not in store:", self.store.keys()))
            raise AttributeError

        try:
            ret = resolve_rna(self, data, name)
            self.__dict__[name] = ret
            return ret
        except:
            print_traceback()
            raise AttributeError(name) from None

    def __dir__(self):
        return self.store.keys()

    def __str__(self):
        return f"<{super().__getattribute__('identifier')}>"


def resolver_as_class(instance: RnaResolver):
    try:
        return return_type_cache[instance]
    except:
        class ReturnType(metaclass=ResolverMeta):
            name = f"Resolver Class {instance.identifier}"
            resolver = RnaResolver(instance, static=True)
            __module__ = "builtins"
            store = resolver.store
        return_type_cache[instance] = ReturnType
    return ReturnType


is_resolver = RnaResolver.__instancecheck__


def stub_from(restype=None, name="stub", vars=(), doc=None):
    def s() -> restype: pass
    s.__name__ = name
    s.__code__ = s.__code__.replace(co_argcount=len(vars), co_varnames=vars)
    s.__doc__ = doc
    return s


def stub_from_func(func, restype=None, vars=()):
    def s() -> restype: pass
    s.__name__ = func.__name__
    s.__code__ = s.__code__.replace(co_argcount=len(vars), co_varnames=vars)
    s.__doc__ = func.__doc__
    return s

def build_context_resolver():
    from types import GenericAlias
    store = RnaResolver(_context).store

    for key, value in screen_context.items():
        if isinstance(value, GenericAlias):
            store[key] = [RnaResolver(value.__args__[0].bl_rna, static=True)]
        else:
            store[key] = RnaResolver(value.bl_rna, static=True)

    store["copy"] = stub_from(dict, "copy")


# Return type mapping for bpy_struct methods - manually edited.
# Not ideal, but bpy_struct does not have a blueprint, and its
# methods are used in all StructRNA instances.
def set_bpy_struct_mapping():
    import typing

    import idprop

    # IDPropertyUIManager isn't exposed, so we do this hack by adding a
    # StringProperty on a bpy.types.ID instance.
    uid = hex(id(utils.this_module()))
    _context.window_manager[uid] = ""
    IDPropertyUIManager = type(_context.window_manager.id_properties_ui(uid))
    _context.window_manager.pop(uid)

    bpy_struct_rtype_mapping = {
        "as_pointer":               int,
        "bl_rna_get_subclass":      bpy_types.Struct,
        "bl_rna_get_subclass_py":   bpy_struct,
        "driver_add":               bpy_types.FCurve | list[bpy_types.FCurve],
        "driver_remove":            bool,
        "get":                      typing.Any,
        "id_properties_clear":      None,
        "id_properties_ensure":     idprop.types.IDPropertyGroup,
        "id_properties_ui":         IDPropertyUIManager, 
        "is_property_hidden":       bool,
        "is_property_overridable_library": bool,
        "is_property_readonly":     bool,
        "is_property_set":          bool,
        "items":                    idprop.types.IDPropertyGroupViewItems,
        "keyframe_delete":          bool,
        "keyframe_insert":          bool,
        "keys":                     idprop.types.IDPropertyGroupViewKeys,
        "path_from_id":             str,
        "path_resolve":             bpy_types.bpy_prop,
        "pop":                      typing.Any,
        "property_overridable_library_set": bool,
        "property_unset":           None,
        "type_recast":              bpy_struct,
        "values":                   idprop.types.IDPropertyGroupViewValues
    }

    for identifier, rtype in bpy_struct_rtype_mapping.items():
        try:
            doc = getattr(bpy_struct, identifier).__doc__
        except AttributeError:
            print("Warning: textension.suggestions bpy_struct API mismatch")
            continue

        def func() -> rtype:
            pass
        func.__doc__ = doc
        setattr(RnaResolver, identifier, func)


# Maps bpy.props functions to their specific resolver.
func_map = {}
type_getattr = type.__getattribute__
_property_deferred_dict = bpy.props._PropertyDeferred.__dict__.copy()
class PropertyDeferredMetaResolver(type):
    __module__ = "builtins"

    def __getattribute__(cls, name):
        if name == "__call__":
            return float
        # print("__getattribute__", cls, name)
        # if name == "__dict__":
        #     return float.__dict__
        return type_getattr(cls, name)


another_map = {
    "StringProperty": str,
    "FloatProperty": float,
    "FloatVectorProperty": None,  # Needs bpy_prop_array

    "IntProperty": int,
    "IntVectorProperty": None,

    "BoolProperty": bool,
    "BoolVectorProperty": None,

    "EnumProperty": str,
    
}

# This represents a _PropertyDeferred class.
class PropResolver(metaclass=PropertyDeferredMetaResolver):
    __module__ = "builtins"
    _deferred_dir = dir(bpy.props._PropertyDeferred)
    _property_types = {}

    # We need __new__ to modify the class on creation, because with a meta
    # resolver's __getattribute__ the class is the first argument and the
    # property information resolved must be unique to the property type.
    def __new__(cls, prop_type, prop_func, is_vector=False):

        prop_name = prop_type.__name__
        
        mapping = {
            "prop_type": prop_type,
            "function": prop_func,
            "prop_name": prop_name,
            "is_vector": is_vector,
            "__name__": f"PropResolver {prop_name}"
        }

        distinct_type = type(
            f"PropResolver {prop_name}",
            (PropResolver,),
            mapping
        )
        return super().__new__(distinct_type)

    def __init__(self, prop_type, prop_func, is_vector=False):
        self.prop_type = prop_type
        self.function = prop_func
        self.is_vector = is_vector
        
        # The names that are allowed to complete when doing:
        # FloatProperty().
        #                 ^
        self.lookup = {"function": prop_func, "keywords": {}}

    def __getattribute__(self, name):
        lookup = super().__getattribute__("lookup")
        # print("PropResolver get", name)
        if name == "__dict__":
            return lookup
        elif name in lookup:
            return lookup[name]
        elif name in dir(self):
            return super().__getattribute__(name)
        elif name == "__annotations__":
            key = super().__getattribute__("prop_name")
            try:
                return {'return': another_map[key]}
            except KeyError:
                print(f"{key} not implemented in PropResolver")
                return None
        raise AttributeError(name)

    # def __repr__(self):
    #     return f"<_PropertyDeferredResolver for {{}}>"

    def __dir__(self):
        return list(super().__getattribute__("lookup").keys())


def build_prop_resolver():

    for prop_type in bpy.types.Property.__subclasses__():
        name = prop_type.bl_rna.identifier
        func = getattr(bpy.props, name)
        func_map[func] = PropResolver(prop_type, func)

        # Vector types aren't distinct from their scalar counterparts,
        # but if bpy.props treats them as such, then we do it.
        vec_name = name.replace("Property", "VectorProperty")
        if vec_fn := getattr(bpy.types, vec_name, None):
            func_map[vec_fn] = PropResolver(prop_type, vec_fn, is_vector=True)


def init_lookups():
    set_bmesh_lookups()
    build_context_resolver()
    build_prop_resolver()