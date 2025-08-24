import time

import torch

from visuomotor.model.resnet import get_resnet, replace_bn_with_gn


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

    # vision_encoder = get_resnet('resnet34')
    # vision_encoder = replace_bn_with_gn(vision_encoder)
    # vision_encoder.to(device)
    vision_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    vision_encoder = replace_bn_with_gn(vision_encoder)
    vision_encoder.eval()
    vision_encoder.to(device)

    # naction = torch.randn((batch_size, pred_horizon, action_dim), device=device)
    # timestep = torch.randn((batch_size, ), device=device)
    # global_cond = torch.randn((batch_size, global_cond_dim), device=device)
    arm_image = torch.randn((batch_size, 1, 3, 224, 224), device=device)

    times = []

    for i in range(n):
        t0 = time.time()
        output = vision_encoder(arm_image.flatten(end_dim=1))
        t1 = time.time()

        times.append(t1 - t0)

        print(t1 - t0)
        print(output.shape)
        assert output.shape == (batch_size, 384)

    print(times)

    times_except_start = times[1:]
    print(sum(times_except_start) / len(times_except_start))
    print(all_parameters(vision_encoder))
    print(trainable_parameters(vision_encoder))
