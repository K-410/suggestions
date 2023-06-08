import gpu


gpu_rtype_data = (
    (gpu.capabilities.extensions_get, tuple[str]),

    (gpu.state.active_framebuffer_get, gpu.types.GPUFrameBuffer),
    (gpu.state.blend_get, str),
    (gpu.state.depth_mask_get, bool),
    (gpu.state.depth_test_get, str),
    (gpu.state.line_width_get, float),
    (gpu.state.scissor_get, tuple[int]),
    (gpu.state.viewport_get, tuple[int]),

    (gpu.shader.from_builtin, gpu.types.GPUShader),
    (gpu.shader.create_from_info, gpu.types.GPUShader),

    (gpu.types.GPUFrameBuffer.viewport_get, tuple[int]),
    (gpu.types.GPUTexture.read, gpu.types.Buffer),
    (gpu.types.GPUShader.attrs_info_get, tuple[tuple[str, str]]),
)


gpu_descriptor_data = (
    (gpu.types.GPUFrameBuffer.is_bound, bool),
    (gpu.types.GPUShader.name, str),
    (gpu.types.GPUShader.program, int),
    (gpu.types.GPUStageInterfaceInfo.name, str),
    (gpu.types.GPUTexture.format, str),
    (gpu.types.GPUTexture.height, int),
    (gpu.types.GPUTexture.width, int),
)


def _apply_mathutils_overrides():
    for obj, rtype in gpu_rtype_data:
        _value_overrides[obj] = _rtype_override
        _rtype_overrides[obj] = rtype

    for descriptor, rtype in gpu_descriptor_data:
        obj = descriptor.__objclass__
        _value_overrides[obj] = _descriptor_override
        _descriptor_overrides[obj][descriptor.__name__] = rtype
