import time

import torch

from visuomotor.config.diffusion_policy_config import DiffusionPolicyConfig
from visuomotor.policy.diffusion_policy import DiffusionPolicy2


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {DEVICE}")
    CONFIG = DiffusionPolicyConfig()

    policy = DiffusionPolicy2(CONFIG, DEVICE)
    with torch.no_grad():
        image = torch.zeros((2, CONFIG.obs_horizon, 3, 96, 96)).to(DEVICE)
        agent_pos = torch.zeros((2, CONFIG.obs_horizon, 2)).to(DEVICE)
        noised_action = policy.initial_noise(2)
        diffusion_iter = torch.zeros((2,)).to(DEVICE)

        t0 = time.time()
        noise = policy.predict_noise(image, agent_pos, noised_action, diffusion_iter)
        t1 = time.time()

        denoised_action = noised_action - noise

        print('time:', t1 - t0)
        print(image.shape)
        print(agent_pos.shape)
        print(denoised_action.shape)