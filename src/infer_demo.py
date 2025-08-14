import os

import torch
import numpy as np
import zarr
from scipy.spatial.transform import Rotation

from visuomotor.dataset.bathroom_dataset import BathroomDataset
from visuomotor.config.diffusion_policy_config import DiffusionPolicy2CamConfig
from visuomotor.policy.diffusion_policy import DiffusionPolicy2Cam
from visuomotor.dataset.tool import normalize_data, unnormalize_data


def symmetric_orthogonalization(x):
    """Maps 9D input vectors onto SO(3) via symmetric orthogonalization.

    x: should have size [batch_size, 9]

    Output has size [batch_size, 3, 3], where each inner 3x3 matrix is in SO(3).
    """
    m = x.view(-1, 3, 3)
    u, s, v = torch.svd(m)
    vt = torch.transpose(v, 1, 2)
    det = torch.det(torch.matmul(u, vt))
    det = det.view(-1, 1, 1)
    vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
    r = torch.matmul(u, vt)
    return r


def main():

    CHECKPOINT = "SD_450_ep_1755147437.ckpt"
    PATH_TO_STORAGE = "/home/andriisydor/masters_thesis/visuomotor_policy/checkpoints"
    PATH_TO_CHECKPOINT = os.path.join(PATH_TO_STORAGE, CHECKPOINT)
    PATH_TO_DATA = "/home/andriisydor/masters_thesis/visuomotor_policy/data/SD_chunked.zarr"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CONFIG = DiffusionPolicy2CamConfig()

    state_dict = torch.load(PATH_TO_CHECKPOINT, map_location='cuda')
    ema_policy = DiffusionPolicy2Cam(CONFIG, DEVICE)
    ema_policy.nets.load_state_dict(state_dict)

    data_split = BathroomDataset.default_dataset_split()
    dataset_root = zarr.open(PATH_TO_DATA, 'r')
    stats = BathroomDataset.calculate_train_stats(dataset_root, data_split["train"])
    print(stats)

    val_dataset = BathroomDataset(
        dataset_root=dataset_root,
        split_indexes=data_split["valid"],
        pred_horizon=16,
        obs_horizon=2,
        action_horizon=8,
        stats=stats
    )

    elem = val_dataset[0]
    torch_elem = {}
    for key, value in elem.items():
        torch_elem[key] = torch.tensor(value).unsqueeze(0)

    arm_images = torch_elem['realsense'].float().to(DEVICE)
    depth_images = torch_elem['depth_camera'].float().to(DEVICE)

    poses = torch.cat([torch_elem["wrist_pos"], torch_elem["wrist_rot"]], dim=2)
    poses = poses.float().to(DEVICE)

    target_actions = torch.cat([torch_elem["action_pos"], torch_elem["action_rot"]], dim=2)
    target_actions = target_actions.float()  # .to(DEVICE)

    print(arm_images.shape, depth_images.shape, poses.shape, target_actions.shape)
    

    B = 1

    # infer action
    with torch.no_grad():
        naction = ema_policy.action(arm_images, depth_images, poses, B)
    

    # unnormalize action
    print(unnormalize_data(elem["wrist_pos"], stats=stats['wrist_pos']))

    naction = naction.detach().to('cpu').numpy()


    action_pos = naction[0, :, :3]  # (pred_horizon, 3)
    action_rot = naction[0, :, 3:]  # (pred_horizon, 9)

    action_pos_pred = unnormalize_data(action_pos, stats=stats['action_pos'])
    action_rot_pred = symmetric_orthogonalization(torch.tensor(action_rot)).numpy()
    print(action_pos_pred)
    print(action_rot_pred)
    print("--------------------------")
    # print(action_rot.reshape(-1, 3, 3) - action_rot_pred)
    quats = Rotation.from_matrix(action_rot_pred).as_quat()
    print(quats)

    # print("----", naction)
    # print("---", target_actions[0, :, 3:])
    # print((torch.tensor(naction) - target_actions) > 0.01)
    # print((torch.tensor(naction) - target_actions))

    # # only take action_horizon number of actions
    # start = CONFIG.obs_horizon - 1
    # end = start + CONFIG.action_horizon
    # action = action_pred[start:end,:]

    # print(action)


if __name__ == "__main__":
    main()
