"""This module adds gpu module inference."""

import gpu


gpu_rtype_data = [
    (gpu.capabilities.extensions_get, tuple[str]),

    (gpu.state.active_framebuffer_get, gpu.types.GPUFrameBuffer),
    (gpu.state.blend_get, str),
    (gpu.state.depth_mask_get, bool),
    (gpu.state.depth_test_get, str),
    (gpu.state.line_width_get, float),
    (gpu.state.viewport_get, tuple[int]),

    (gpu.shader.from_builtin, gpu.types.GPUShader),
    (gpu.shader.create_from_info, gpu.types.GPUShader),

    (gpu.types.GPUFrameBuffer.viewport_get, tuple[int]),
    (gpu.types.GPUTexture.read, gpu.types.Buffer),
]


# Versioning.
if _func := getattr(gpu.state, "scissor_get", None):
    gpu_rtype_data += (_func, tuple[int]),


if _func := getattr(gpu.types, "attrs_info_get", None):
    gpu_rtype_data += (_func, tuple[tuple[str, str]]),


gpu_descriptor_data = (
    (gpu.types.GPUFrameBuffer.is_bound, bool),
    (gpu.types.GPUShader.name, str),
    (gpu.types.GPUShader.program, int),
    (gpu.types.GPUStageInterfaceInfo.name, str),
    (gpu.types.GPUTexture.format, str),
    (gpu.types.GPUTexture.height, int),
    (gpu.types.GPUTexture.width, int),
)


def apply():
    from .. import tools

    tools._add_rtype_overrides(gpu_rtype_data)
    tools._add_descriptor_overrides(gpu_descriptor_data)
