import time

import zarr
import torch

from ..src.visuomotor.dataset.push_t_dataset import PushTImageDataset


def main():
    PATH_TO_DATA = "/home/andriisydor/masters_thesis/visuomotor_policy/data/pusht_cchi_v7_replay.zarr"
    BATCH_SIZE = 64

    data_split = PushTImageDataset.default_dataset_split()
    dataset_root = zarr.open(PATH_TO_DATA, 'r')
    stats = PushTImageDataset.calculate_train_stats(dataset_root, data_split["train"])

    for split_type, split_indexes in data_split.items():
        
        dataset = PushTImageDataset(
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
        print("batch['image'].shape:", batch['image'].shape)
        print("batch['agent_pos'].shape:", batch['agent_pos'].shape)
        print("batch['action'].shape", batch['action'].shape)
        print()


if __name__ == "__main__":
    main()
