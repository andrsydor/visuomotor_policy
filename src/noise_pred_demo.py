import time

import torch

from visuomotor.model.unet1d import ConditionalUnet1D


def trainable_parameters(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params


def all_parameters(model):
    num_params = sum(p.numel() for p in model.parameters())
    return num_params


if __name__ == "__main__":

    pred_horizon = 16
    action_dim = 12
    global_cond_dim = 1048
    batch_size = 1
    n = 100

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=global_cond_dim
    )

    noise_pred_net.to(device)

    naction = torch.randn((batch_size, pred_horizon, action_dim), device=device)
    timestep = torch.randn((batch_size, ), device=device)
    global_cond = torch.randn((batch_size, global_cond_dim), device=device)

    times = []

    for i in range(n):
        t0 = time.time()
        output = noise_pred_net(naction, timestep, global_cond)
        t1 = time.time()

        times.append(t1 - t0)

        print(t1 - t0)
        print(output.shape)
        assert output.shape == (batch_size, pred_horizon, action_dim)

    print(times)

    times_except_start = times[1:]
    print(sum(times_except_start) / len(times_except_start))
    print(all_parameters(noise_pred_net))
    print(trainable_parameters(noise_pred_net))
