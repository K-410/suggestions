from mathutils import Vector, Quaternion, Euler, Matrix, Color
from itertools import product, repeat, chain

from ..tools import _add_rtype_overrides, _add_descriptor_overrides



# Blender FloatProperty subtype-to-mathutils type mapping.
float_vector_map = {
    "ACCELERATION": Vector,
    "DIRECTION":    Vector,
    "TRANSLATION":  Vector,
    "VELOCITY":     Vector,
    "XYZ_LENGTH":   Vector,
    "XYZ":          Vector,
    "COORDINATES":  Vector,

    "MATRIX":       Matrix,
    "EULER":        Euler,
    "QUATERNION":   Quaternion,
    "COLOR":        Color,
    "COLOR_GAMMA":  Color
}


vector_rtype_data = (
    (Vector.Fill, Vector),
    (Vector.Linspace, Vector),
    (Vector.Range, Vector),
    (Vector.Repeat, Vector),
    (Vector.copy, Vector),
    (Vector.cross, Vector | float),  # type:ignore
    (Vector.dot, float),
    (Vector.freeze, Vector),
    (Vector.lerp, Vector),
    (Vector.normalized, Vector),
    (Vector.orthogonal, Vector),
    (Vector.project, Vector),
    (Vector.reflect, Vector),
    (Vector.resized, Vector),
    (Vector.rotation_difference, Quaternion),
    (Vector.slerp, Vector),
    (Vector.to_2d, Vector),
    (Vector.to_3d, Vector),
    (Vector.to_4d, Vector),
    (Vector.to_track_quat, Quaternion),
    (Vector.to_tuple, tuple[float]),
    (Vector.__copy__, Vector),
    (Vector.__deepcopy__, float),
    (Vector.__getitem__, float),
    (Vector.__matmul__, float),
)
vector_descriptor_data = [
    (Vector.x, float),
    (Vector.y, float),
    (Vector.z, float),
    (Vector.w, float),
]
euler_rtype_data = (
    (Euler.__getitem__, float),
    (Euler.to_matrix, Matrix),
    (Euler.to_quaternion, Quaternion),
    (Euler.copy, Euler),
    (Euler.__copy__, Euler),
    (Euler.__deepcopy__, Euler),
)
euler_descriptor_data = (
    (Euler.x, float),
    (Euler.y, float),
    (Euler.z, float),
    (Euler.order, str),
    (Euler.is_wrapped, bool),
    (Euler.is_frozen, bool),
    (Euler.is_valid, bool),
)



def apply():
    # Add swizzle descriptor overrides.
    for i in (2, 3, 4):
        swizzles = map("".join, product("xyzw", repeat=i))
        vector_descriptor_data.extend(
            zip(map(Vector.__dict__.__getitem__, swizzles), repeat(Vector)))

    _add_rtype_overrides(vector_rtype_data)
    _add_descriptor_overrides(vector_descriptor_data)

    _add_rtype_overrides(euler_rtype_data)
    _add_descriptor_overrides(euler_descriptor_data)
