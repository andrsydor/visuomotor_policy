import time
import argparse

import zarr
import torch

from visuomotor.dataset.push_t_dataset import PushTImageDataset
from visuomotor.task.components import task_from_string


def main():
    parser = argparse.ArgumentParser(description="Dataset demo")
    parser.add_argument("--dataset", help="Path to the zarr file")  # "/home/andriisydor/masters_thesis/visuomotor_policy/data/pusht_cchi_v7_replay.zarr"
    parser.add_argument("--task", help="Task name")  # "push_t"
    args = parser.parse_args()

    PATH_TO_DATA = args.dataset
    TASK_NAME = args.task
    BATCH_SIZE = 64

    task_class = task_from_string(TASK_NAME)
    dataset_class = task_class.dataset_class()

    data_split = dataset_class.default_dataset_split()
    dataset_root = zarr.open(PATH_TO_DATA, 'r')
    stats = dataset_class.calculate_train_stats(dataset_root, data_split["train"])

    for split_type, split_indexes in data_split.items():

        dataset = dataset_class(
            dataset_root=dataset_root,
            split_indexes=split_indexes,
            pred_horizon=16,
            obs_horizon=2,
            action_horizon=8,
            stats=stats)
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=1,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True
        )

        t0 = time.time()
        batch = next(iter(dataloader))
        t1 = time.time()
        print(f"--- {split_type} ---")
        print(t1 - t0)
        print("image.shape:", batch['image'].shape)
        print("agent_pos.shape:", batch['agent_pos'].shape)
        print("action.shape", batch['action'].shape)
        print()


if __name__ == "__main__":
    main()
