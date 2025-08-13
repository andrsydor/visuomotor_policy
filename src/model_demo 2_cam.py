import time

import torch

from visuomotor.config.diffusion_policy_config import DiffusionPolicy2CamConfig
from visuomotor.policy.diffusion_policy import DiffusionPolicy2Cam


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {DEVICE}")
    CONFIG = DiffusionPolicy2CamConfig()

    policy = DiffusionPolicy2Cam(CONFIG, DEVICE)
    with torch.no_grad():
        arm_image = torch.zeros((2, CONFIG.image_obs_horizon, 3, 224, 224)).to(DEVICE)
        depth_image = torch.zeros((2, CONFIG.image_obs_horizon, 3, 224, 224)).to(DEVICE)
        agent_pos = torch.zeros((2, CONFIG.obs_horizon, 2)).to(DEVICE)
        noised_action = policy.initial_noise(2)
        diffusion_iter = torch.zeros((2,)).to(DEVICE)

        t0 = time.time()
        noise = policy.predict_noise(arm_image, depth_image, agent_pos, noised_action, diffusion_iter)
        t1 = time.time()

        denoised_action = noised_action - noise

        print('time:', t1 - t0)
        print(arm_image.shape)
        print(depth_image.shape)
        print(agent_pos.shape)
        print(denoised_action.shape)