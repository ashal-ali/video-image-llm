# The names of CLIP layers (for freezing)
CLIP_INIT_LAYERS = [
    "patch_embed",
    "norm_pre",
    "norm1",
    "attn",
    "drop_path",
    "norm2",
    "mlp",
    "norm3", # CLIP doesn't have norm3, but we add it so that the ViT output stays unchanged during training
    "norm",
    "cls_token",
    "pos_embed",
]
