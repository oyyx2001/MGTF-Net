"""Formal ungated, four-branch MGTF-Net with continuous AR decoding."""
import torch
from torch import nn
from .graph_attention import SceneGATv2
from .temporal_encoder import make_temporal_encoder
from .decoder import decode


class MGTFNet(nn.Module):
    def __init__(self, obs_len=12, pred_len=20, width=64, dropout=0.1, normalization=None):
        super().__init__()
        self.obs_len, self.pred_len = obs_len, pred_len
        defaults = {"rel_mean": torch.zeros(6), "rel_std": torch.ones(6),
                    "pos_mean": torch.zeros(3), "pos_std": torch.ones(3)}
        for name, value in defaults.items():
            self.register_buffer(name, (normalization or {}).get(name, value).float().clone())
        self.input_projection = nn.Sequential(nn.Linear(9, width), nn.LayerNorm(width), nn.GELU())
        self.time_embedding = nn.Parameter(torch.randn(obs_len, 1, width) * 0.02)
        self.trajectory_lstm = nn.LSTM(width, width)
        self.graph_attention = SceneGATv2(width, 4)
        self.graph_lstm = nn.LSTM(width, width)
        self.temporal_transformer = make_temporal_encoder(width, dropout)
        self.history_projection = nn.Sequential(nn.Linear(obs_len * 9, width), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(width * 4, width * 2), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(width * 2, width), nn.LayerNorm(width))
        self.decoder_lstm = nn.LSTMCell(width + 4, width)
        self.decoder = nn.Linear(width, 3)
        nn.init.normal_(self.decoder.weight, std=0.001)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, obs_rel, obs_pos, boundaries):
        """Native interface: [12,N,6] km/km-per-second, [12,N,3] km.

        Return [20,N,6]; only the first three channels are displacements.
        Boundaries are CPU [scenes,2] start/end indices, never cross-scene edges.
        """
        if obs_rel.shape[0] != self.obs_len or obs_pos.shape[0] != self.obs_len:
            raise ValueError("Pass observed timesteps only, never concatenate the future")
        rel = (obs_rel - self.rel_mean) / self.rel_std
        pos = (obs_pos[:, :, :3] - self.pos_mean) / self.pos_std
        features = torch.cat((rel, pos), -1)
        embedded = self.input_projection(features) + self.time_embedding
        motion, _ = self.trajectory_lstm(embedded)
        graph = self.graph_attention(motion, pos, boundaries)
        graph, _ = self.graph_lstm(graph)
        temporal = self.temporal_transformer(embedded.transpose(0, 1))[:, -1]
        history = self.history_projection(features.permute(1, 0, 2).reshape(features.shape[1], -1))
        context = self.fusion(torch.cat((motion[-1], graph[-1], temporal, history), -1))
        xyz = decode(context, obs_rel, self.rel_std, self.pred_len, self.decoder_lstm, self.decoder)
        return torch.cat((xyz, obs_rel[-1:, :, 3:].expand(self.pred_len, -1, -1)), -1)

    def predict_positions(self, observed_features, scene_ids):
        """Public metric-unit interface: [N,12,9] -> [N,20,3] meters."""
        from utils.preprocessing import to_core_inputs
        rel, pos, boundaries = to_core_inputs(observed_features, scene_ids)
        increments = self(rel, pos, boundaries)[:, :, :3]
        return (increments.cumsum(0) + pos[-1]).transpose(0, 1) * 1000.0


def multi_horizon_loss(pred_rel, obs_pos, target, horizons):
    """Formal weighted ADE + 0.5 FDE objective, with native positions in km."""
    pred_pos = pred_rel[:, :, :3].cumsum(0) + obs_pos[-1, :, :3]
    error = pred_pos - target[:, :, :3]
    distances = (error.square().sum(-1) + 1e-8).sqrt()
    return torch.stack([
        (distances[:steps].mean() + 0.5 * distances[steps - 1].mean()) * (100.0 / int(seconds))
        for seconds, steps in horizons.items()
    ]).mean()
