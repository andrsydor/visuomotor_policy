import os
import collections

import torch
import numpy as np
import zarr
from skvideo.io import vwrite

from visuomotor.dataset.push_t_dataset import PushTImageDataset
from visuomotor.config.diffusion_policy_config import DiffusionPolicyConfig
from visuomotor.policy.diffusion_policy import DiffusionPolicy2
from visuomotor.env.push_t import PushTImageEnv
from visuomotor.dataset.tool import normalize_data, unnormalize_data


def main():

    CHECKPOINT = "DP_ema_100_ep_100_data_1753629063.ckpt"
    PATH_TO_STORAGE = "/home/andriisydor/masters_thesis/visuomotor_policy/checkpoints"
    PATH_TO_VIDEOS = "/home/andriisydor/masters_thesis/visuomotor_policy/video"
    PATH_TO_CHECKPOINT = os.path.join(PATH_TO_STORAGE, CHECKPOINT)
    PATH_TO_DATA = "/home/andriisydor/masters_thesis/visuomotor_policy/data/pusht_cchi_v7_replay.zarr"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CONFIG = DiffusionPolicyConfig()

    state_dict = torch.load(PATH_TO_CHECKPOINT, map_location='cuda')
    ema_policy = DiffusionPolicy2(CONFIG, DEVICE)
    ema_policy.nets.load_state_dict(state_dict)

    data_split = PushTImageDataset.default_dataset_split()
    dataset_root = zarr.open(PATH_TO_DATA, 'r')
    stats = PushTImageDataset.calculate_train_stats(dataset_root, data_split["train"])

    seeds = list(range(100000, 100002))
    episode_rewards = []
    for seed in seeds:
        # limit enviornment interaction to 200 steps before termination
        max_steps = 200
        env = PushTImageEnv()
        # use a seed >200 to avoid initial states seen in the training dataset
        env.seed(seed)

        # get first observation
        obs, info = env.reset()

        # keep a queue of last 2 steps of observations
        obs_deque = collections.deque([obs] * CONFIG.obs_horizon, maxlen=CONFIG.obs_horizon)
        # save visualization and rewards
        imgs = [env.render(mode='rgb_array')]
        rewards = list()
        done = False
        step_idx = 0

        while not done:
            B = 1
            # stack the last obs_horizon number of observations
            images = np.stack([x['image'] for x in obs_deque])
            agent_poses = np.stack([x['agent_pos'] for x in obs_deque])

            nagent_poses = normalize_data(agent_poses, stats=stats['agent_pos'])
            nimages = images

            nimages = torch.from_numpy(nimages).to(DEVICE, dtype=torch.float32).unsqueeze(0)
            nposes = torch.from_numpy(nagent_poses).to(DEVICE, dtype=torch.float32).unsqueeze(0)
            # print(nimages.size(), nposes.size())

            # infer action
            with torch.no_grad():
                naction = ema_policy.action(nimages, nposes, B)

            # unnormalize action
            naction = naction.detach().to('cpu').numpy()
            # (B, pred_horizon, action_dim)
            naction = naction[0]
            action_pred = unnormalize_data(naction, stats=stats['action'])

            # only take action_horizon number of actions
            start = CONFIG.obs_horizon - 1
            end = start + CONFIG.action_horizon
            action = action_pred[start:end,:]
            # (action_horizon, action_dim)

            # execute action_horizon number of steps
            # without replanning
            for i in range(len(action)):
                # stepping env
                obs, reward, done, _, info = env.step(action[i])
                # save observations
                obs_deque.append(obs)
                # and reward/vis
                rewards.append(reward)
                imgs.append(env.render(mode='rgb_array'))

                # update progress bar
                step_idx += 1
                if step_idx > max_steps:
                    done = True
                if done:
                    break

        # print out the maximum target coverage
        max_reward = max(rewards)
        print('Score: ', max_reward)
        episode_rewards.append(max_reward)

        video_name = f"eval_{seed}.mp4"
        path_to_video = os.path.join(PATH_TO_VIDEOS, video_name)
        vwrite(path_to_video, imgs)


if __name__ == "__main__":
    main()
