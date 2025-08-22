import os

import zarr
from PIL import Image
import numpy as np
import json
from tqdm import tqdm

from visuomotor.utils.fk import FK


def convert(path_to_dataset: str, ouput_file_path: str) -> None:

    if os.path.exists(ouput_file_path):
        raise ValueError("Dataset already exists!")

    image_names = ["realsense.jpg", "depth_camera.jpg", "front_bottom_camera.jpg"]
    json_name = "robot_state.json"

    height, width, channels = 224, 224, 3

    fk = FK()

    root = zarr.open(ouput_file_path, mode="w")
    data_group = root.create_group("data")

    camera_arrays = []

    for camera_i, image_name in enumerate(image_names):
        camera_name = os.path.splitext(image_name)[0]
        arr = data_group.create_dataset(
            name=camera_name,
            shape=(0, channels, height, width),
            dtype="float32"
        )
        camera_arrays.append(arr)

    joint_array = data_group.create_dataset(
        name="joints",
        shape=(0, 7),
        dtype="float32"
    )

    wrist_pos_array = data_group.create_dataset(
        name="wrist_pos",
        shape=(0, 3),
        dtype="float32"
    )

    wrist_rot_array = data_group.create_dataset(
        name="wrist_rot",
        shape=(0, 9),
        dtype="float32"
    )

    dataset_dir_elems = os.listdir(path_to_dataset)
    demo_dirs = [
        elem_name for elem_name in dataset_dir_elems if os.path.isdir(os.path.join(path_to_dataset, elem_name))
    ]
    demo_dirs = sorted(demo_dirs)

    episode_ends = []

    for demo_dir in tqdm(demo_dirs, desc="Demo dirs"):

        demo_dir_path = os.path.join(path_to_dataset, demo_dir)
        sample_dirs = sorted(os.listdir(demo_dir_path))

        for sample_dir in sample_dirs:
            sample_dir_path = os.path.join(demo_dir_path, sample_dir)

            for camera_i, image_name in enumerate(image_names):
                image_path = os.path.join(sample_dir_path, image_name)
                pil_image = Image.open(image_path).resize((width, height), resample=Image.Resampling.BICUBIC)
                np_image = np.array(pil_image)
                np_image = np.transpose(np_image, (2, 0, 1))
                np_image = np_image / 255.0
                np_image = np.expand_dims(np_image, axis=0)
                camera_arrays[camera_i].append(np_image)
            
            state_json_path = os.path.join(sample_dir_path, json_name)
            with open(state_json_path, "r") as state_json_file:
                state_json_dict = json.load(state_json_file)
                joint_angles = state_json_dict["robotState"]["armState"]["joints"]["values"]
                np_joint_pos = joint_angles
                np_joint_pos = np.expand_dims(np_joint_pos, axis=0)
                joint_array.append(np_joint_pos)
                
                wrist_pos, wrist_rot = fk(joint_angles)
                
                np_wrist_pos = np.expand_dims(wrist_pos, axis=0)
                wrist_pos_array.append(np_wrist_pos)

                np_wrist_rot = np.expand_dims(wrist_rot.as_matrix().flatten(), axis=0)
                wrist_rot_array.append(np_wrist_rot)
        
        assert camera_arrays[0].shape[0] == joint_array.shape[0]
        episode_ends.append(joint_array.shape[0])

    assert len(demo_dirs) == len(episode_ends)

    meta_group = root.create_group("meta")
    np_episode_ends_array = np.array(episode_ends)
    meta_group.create_dataset("episode_ends", data=np_episode_ends_array)
