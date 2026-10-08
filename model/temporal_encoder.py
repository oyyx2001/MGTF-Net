"""Two-layer temporal encoder with independent clone initialization."""
from torch import nn


def make_temporal_encoder(width, dropout):
    layer = nn.TransformerEncoderLayer(width, 4, width * 2, dropout,
                                       activation="gelu", batch_first=True, norm_first=True)
    encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
    for block in encoder.layers:
        nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
        nn.init.xavier_uniform_(block.self_attn.out_proj.weight)
        nn.init.xavier_uniform_(block.linear1.weight)
        nn.init.xavier_uniform_(block.linear2.weight)
    return encoder
