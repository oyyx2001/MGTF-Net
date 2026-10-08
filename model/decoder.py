"""Continuous autoregressive displacement decoder used by the formal model."""
import torch


def decode(context, obs_rel, rel_std, pred_len, decoder_lstm, decoder):
    hidden, cell = context, torch.zeros_like(context)
    velocity = obs_rel[-1, :, :3]
    trend = obs_rel[-4:, :, :3].mean(0)
    future = []
    for t in range(pred_len):
        elapsed = context.new_full((context.shape[0], 1), (t + 1) / pred_len)
        inputs = torch.cat((context, velocity / rel_std[:3], elapsed), -1)
        hidden, cell = decoder_lstm(inputs, (hidden, cell))
        velocity = trend + decoder(hidden) * rel_std[:3]
        future.append(velocity)
    return torch.stack(future)
