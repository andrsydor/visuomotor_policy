import time

import torch
from torch import nn

from visuomotor.model.transformer import ActionTransformer
from visuomotor.model.tools import get_optim_groups, init_transformer_weights


def trainable_parameters(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params


def all_parameters(model):
    num_params = sum(p.numel() for p in model.parameters())
    return num_params


if __name__ == "__main__":

    pred_horizon = 16
    action_dim = 12
    cond_emb_dim = 512
    batch_size = 32
    n = 100

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    model = ActionTransformer(16, 12, 4, nlayer=3, nhead=4)
    model.to(device)

    naction = torch.randn((batch_size, pred_horizon, action_dim), device=device)
    img_emb = torch.randn((batch_size, 2, 512), device=device)
    agent_pos = torch.randn((batch_size, 2, 12), device=device)

    times = []

    for i in range(n):
        t0 = time.time()
        output = model(img_emb, agent_pos)
        t1 = time.time()

        times.append(t1 - t0)

        print(t1 - t0)
        print(output.shape, naction.shape)
        assert output.shape == naction.shape

    print(get_optim_groups(model))
    print(times)

    times_except_start = times[1:]
    print(sum(times_except_start) / len(times_except_start))
    print(all_parameters(model))
    print(trainable_parameters(model))
