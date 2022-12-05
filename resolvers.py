import sys
import bpy
from types import BuiltinMethodType
from bpy import types as bpy_types
from ... import types, utils
from typing import List, Tuple

_context = utils._context
bpy_struct = bpy_types.bpy_struct

ANSI_YELLOW     = '\033[33m'
ANSI_RED        = '\033[31m'
ANSI_GREEN      = '\033[32m'
ANSI_DEFAULT    = '\033[0m'


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
default_code = types.noop.__code__.replace
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

def dbg(*args):
    f = sys._getframe(1)
    depth = 0
    while f is not None:
        if "self" in f.f_locals and is_resolver(f.f_locals["self"]):
            break
        depth += 1
        f = f.f_back
    else:
        depth = 0
    if depth:
        args = ((" " * depth),) + args
    print(*args)


def stub_from_rna_func(rna, fn):
    try:
        return stub_cache[rna, fn]
    except:
        pass

    dbg(green("creating stub"))
    # TODO: support is_required, is_never_none, is_argument_optional
    param = {}
    for p in fn.parameters:
        if p.is_output:
            dbg("  found output", p)
            ret_rna = p.fixed_type

            # Default is bpy.types.Object.copy() -> bpy.types.ID.
            # This makes the return type the same as the parent rna.
            if fn.identifier == "copy" and ret_rna == struct_id:
                ret_rna = rna

            # Jedi expects the return type to be a class so we have to
            # use metaclasses to provide resolution for return types.
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
            dbg("  found param", p)
            param[p.identifier] = None

    ret = param.pop("return", None)
    stub = stub_from(ret, fn.identifier, tuple(param), fn.description)
    stub_cache[rna, fn] = stub
    return stub


def resolve_rna(resolver, rna, attr):
    if isinstance(rna, bpy_types.Property):
        rna_type = rna.type

        if rna_type == 'POINTER':  # Pointer to an RNA type
            dbg("is a pointer")
            ret = RnaResolver(rna.fixed_type, static=True)

        elif rna_type == 'COLLECTION':  # Pointer to a property collection
            dbg("is a Collection")
            ret = CollectionResolver(rna, resolver)
        else:
            dbg("is a Pytype")
            ret = rna_pytypes[rna_type]

    elif isinstance(rna, bpy_types.Function):
        dbg("is a Function")
        ret = stub_from_rna_func(resolver, rna)

    elif isinstance(rna, (types.Callable, BuiltinMethodType)):
        dbg("is a Method")
        ret = rna

    # Dynamic additions like context resolver
    elif isinstance(rna, RnaResolver):
        dbg("is a Resolver")
        ret = rna

    else:
        # raise AttributeError(f"unhandled: {attr} {rna} {type(rna)}")
        dbg(yellow(f"unhandled: {attr}", rna, type(rna)))
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
    __class__ = list  # Required for subscript/iterator
    __module__ = "builtins"
    def __new__(cls, rna, data):
        # The rna type expected
        assert isinstance(rna, bpy.types.CollectionProperty)

        try:
            return collection_resolver_cache[rna]
        except:
            pass

        # print("Creating CollectionResolver")
        self = collection_resolver_cache.setdefault(rna, super().__new__(cls))
        self.store = store = collection_dict.copy()

        # XXX: id_data is difficult to resolve without walking bpy_types and
        # visiting all rna types and map every collection. Blender simply
        # doesn't store their refined type in bl_rna.
        self.data = data

        srna = rna.srna
        if srna is not None:
            # Resolver for this collection
            resolver = RnaResolver(srna, static=True)
            resolver.store.update(store)
            store.update(srna.properties.items() + srna.functions.items())
            store["bl_rna"] = rna
            self.resolver = resolver
        else:
            print(red("srna is None for", rna))
            # raise AssertionError(red("srna is None for", rna))

        # Resolver for the element type in the collection
        self.restype = restype = resolver_as_class(RnaResolver(rna.fixed_type, static=True))

        # Required for 'for loops'
        self.__iter__ = stub_from(restype, "__iter__")
        return self

    def __getattr__(self, name):
        print("getting", repr(name))

        # Static resolution for bpy_prop_collection methods
        if name == "get":
            return stub_from(self.restype)
        elif name == "items":
            return stub_from(List[Tuple[str, self.restype]])
        elif name == "values":
            return stub_from(List[self.restype])

        try:
            ret = getattr(self.resolver, name)
            # ret = self.store[name]
        except:
            print("failed getting", name)
            raise AttributeError
        else:
            try:
                ret = resolve_rna(None, ret, name)
                print(1)
                return ret
            except:
                ret = getattr(self.resolver, name)
                print(2)
                return ret
            if isinstance(ret, bpy_struct):
                ret = RnaResolver(ret, static=True)
                ret = getattr(ret, name)
                print("return bl_rna", ret)
                return ret

            if is_getset_descriptor(ret):
                print("is getset descriptor", name, ret)
                if name == "data":
                    print("returning data")
                    return self.data

    # Required for subscript
    def __iter__(self):
        return iter((self.restype,))

    def __dir__(self):
        return list(self.store.keys())


class RnaResolver:
    """RnaResolver provides resolutions for RNA instances by listing bl_rna
    methods and properties and resolving them as instance attributes in order
    to provide completions.
    """

    store: dict
    # __objclass__ = None  # XXX: Is this needed?

    def __new__(cls, rna, *, static=False):
        if is_resolver(rna) or (not static and rna.as_pointer() == rna.bl_rna.as_pointer()):
            return rna

        try:
            return resolver_cache[rna.bl_rna]
        except KeyError:
            bl_rna = rna.bl_rna
            self = resolver_cache.setdefault(bl_rna, super().__new__(cls))
            self.identifier = bl_rna.identifier

            self.store = store = bpy_struct.__dict__ | type(rna).__dict__
            store.update(bl_rna.properties.items() + bl_rna.functions.items())
            return self

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

