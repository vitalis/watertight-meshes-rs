"""Cast Trellis2 VAE block weights to FP8 after VAELoader.

Comfy's loader finishes with ``.to(vae_dtype)``, which would undo a cast
done inside the VAE constructor. This node runs on the loaded ``VAE`` and
stores ``float8_e4m3fn`` only on the block convolutions and linears.
``from_latent``, ``output_layer``, norms, and the structure decoder's
``input_layer`` and ``out_layer`` stay on the loader dtype.

``SparseConv3d`` upcasts its weight in forward. ``ops.Linear`` and
``ops.Conv3d`` do that only when ``comfy_cast_weights`` is set, so the
cast turns that flag on for those modules.
"""

import logging

import torch
import torch.nn as nn

FP8 = torch.float8_e4m3fn

SHAPE_ROOTS = (
    ("shape_dec", "blocks"),
    ("struct_dec", "middle_block"),
    ("struct_dec", "blocks"),
)
TEXTURE_ROOTS = (("txt_dec", "blocks"),)

log = logging.getLogger("trellis2_vae_fp8")


def _castable(module):
    return (
        isinstance(module, (nn.Linear, nn.Conv3d))
        or type(module).__name__ == "SparseConv3d"
    )


def _cast_parameter(param):
    if param is None or not param.is_floating_point() or param.dtype == FP8:
        return False
    with torch.no_grad():
        param.data = param.detach().to(dtype=FP8)
    return True


def cast_module_tree(root):
    """Cast conv and linear parameters under ``root``. Returns the parameter count."""
    count = 0
    for module in root.modules():
        if not _castable(module):
            continue
        if hasattr(module, "comfy_cast_weights"):
            module.comfy_cast_weights = True
        for name in ("weight", "bias"):
            if _cast_parameter(getattr(module, name, None)):
                count += 1
    return count


def _roots(model):
    kind = type(model).__name__
    if kind == "ShapeVae":
        spec = SHAPE_ROOTS
    elif kind == "TextureVae":
        spec = TEXTURE_ROOTS
    else:
        raise ValueError(f"Trellis2VaeFp8 expected ShapeVae or TextureVae, got {kind}")
    roots = []
    for parent_name, child_name in spec:
        parent = getattr(model, parent_name)
        roots.append(getattr(parent, child_name))
    return roots


def cast_trellis2_vae(model):
    total = 0
    for root in _roots(model):
        total += cast_module_tree(root)
    return total


class Trellis2VaeFp8:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae": ("VAE",)}}

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "cast"
    CATEGORY = "trellis2"
    DESCRIPTION = "Store Trellis2 VAE block conv and linear weights as float8_e4m3fn. Place after VAELoader."

    def cast(self, vae):
        model = getattr(vae, "first_stage_model", None)
        if model is None:
            raise ValueError("Trellis2VaeFp8 expected a loaded VAE")
        count = cast_trellis2_vae(model)
        patcher = getattr(vae, "patcher", None)
        patched = getattr(patcher, "model", None) if patcher is not None else None
        if patched is not None and patched is not model:
            count += cast_trellis2_vae(patched)
        log.info(
            "Trellis2 VAE FP8 cast %s parameters on %s", count, type(model).__name__
        )
        return (vae,)


NODE_CLASS_MAPPINGS = {
    "Trellis2VaeFp8": Trellis2VaeFp8,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Trellis2VaeFp8": "Trellis2 VAE FP8",
}
