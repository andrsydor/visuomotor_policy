import os

import zarr
import matplotlib.pyplot as plt
import numpy as np


# dataset = zarr.open("/home/andriisydor/masters_thesis/visuomotor_policy/data/pusht_cchi_v7_replay.zarr", "r")
# dataset = zarr.open("/home/andriisydor/masters_thesis/visuomotor_policy/data/IsaacLab_SD_08-08-2025-reviewed.zarr", "r")
# dataset = zarr.open("/home/andriisydor/masters_thesis/visuomotor_policy/data/reviewed-small-images.zarr", "r")
# dataset = zarr.open("/home/andriisydor/masters_thesis/visuomotor_policy/data/sd_dataset_67_ep.zarr", "r")
dataset = zarr.open("/home/andriisydor/masters_thesis/visuomotor_policy/data/SD_fixed_67_ep.zarr", "r")


print([i for i in dataset.group_keys()])
print([i for i in dataset["meta"].array_keys()])
print(dataset["data"]["realsense"].shape)
print(dataset["data"]["depth_camera"].shape)
print(dataset["data"]["front_bottom_camera"].shape)
print(dataset["data"]["joints"].shape)
print(dataset["data"]["wrist_pos"].shape)
print(dataset["data"]["wrist_rot"].shape)
print(list(dataset["meta"]["episode_ends"]))

print(dataset["data"]["wrist_pos"][-1:].shape)
print(np.concat((dataset["data"]["wrist_pos"][1:], dataset["data"]["wrist_pos"][-1:]), axis=0).shape)

root = dataset

# Access arrays
images = root["data/realsense"]           # shape: (N, H, W, C)
joint_positions = root["data/joints"]  # shape: (N, num_joints)
wrist_pos_positions = root["data/wrist_pos"]
wrist_rot_positions = root["data/wrist_rot"]
episode_ends = root["meta/episode_ends"]  # shape: (num_episodes,)

print("Images shape:", images.shape)
print("Joints shape:", joint_positions.shape)
print("Episode ends:", episode_ends[:])

# Example: visualize first episode
start = 224
end = 228

for i in range(start, end):
    frame = np.transpose(images[i], (1, 2, 0))
    joints = joint_positions[i]
    wrist_pos = wrist_pos_positions[i]
    wrist_rot = wrist_rot_positions[i]

    plt.imshow(frame)
    plt.title(f"Frame {i}, joints: {np.round(joints, 2)}\nwrist_pos: {np.round(wrist_pos, 2)}\nwrist_rot: {np.round(wrist_rot, 2)}")
    plt.axis('off')
    plt.show()
