import os

import zarr
import matplotlib.pyplot as plt
import numpy as np


# dataset = zarr.open("/home/andriisydor/masters_thesis/visuomotor_policy/data/pusht_cchi_v7_replay.zarr", "r")
# dataset = zarr.open("/home/andriisydor/masters_thesis/visuomotor_policy/data/IsaacLab_SD_08-08-2025-reviewed.zarr", "r")
# dataset = zarr.open("/home/andriisydor/masters_thesis/visuomotor_policy/data/reviewed-small-images.zarr", "r")
dataset = zarr.open("/home/andriisydor/masters_thesis/visuomotor_policy/data/sd_dataset_67_ep.zarr", "r")


print([i for i in dataset.group_keys()])
print([i for i in dataset["meta"].array_keys()])
print(dataset["data"]["realsense"].shape)
print(dataset["data"]["depth_camera"].shape)
print(dataset["data"]["front_bottom_camera"].shape)
print(dataset["data"]["joints"].shape)
print(dataset["data"]["wrist"].shape)
print(list(dataset["meta"]["episode_ends"]))

print(dataset["data"]["wrist"][-1:].shape)
print(np.concat((dataset["data"]["wrist"][1:], dataset["data"]["wrist"][-1:]), axis=0).shape)

# root = dataset

# # Access arrays
# images = root["data/realsense"]           # shape: (N, H, W, C)
# joint_positions = root["data/joints"]  # shape: (N, num_joints)
# wrist_positions = root["data/wrist"]  # shape: (N, num_joints)
# episode_ends = root["meta/episode_ends"]  # shape: (num_episodes,)

# print("Images shape:", images.shape)
# print("Joints shape:", joint_positions.shape)
# print("Episode ends:", episode_ends[:])

# # Example: visualize first episode
# start = 140
# end = 228

# for i in range(start, end):
#     frame = images[i]
#     joints = joint_positions[i]
#     wrist = wrist_positions[i]

#     plt.imshow(frame)
#     plt.title(f"Frame {i}, joints: {np.round(joints, 2)}\nwrist: {np.round(wrist, 2)}")
#     plt.axis('off')
#     plt.show()

