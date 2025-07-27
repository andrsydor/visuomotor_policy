import time

import torch
from torch import nn
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
import zarr
import numpy as np
from tqdm.auto import tqdm

from visuomotor.dataset.push_t_dataset import PushTImageDataset
from visuomotor.config.diffusion_policy_config import DiffusionPolicyConfig
from visuomotor.policy.diffusion_policy import DiffusionPolicy2
from visuomotor.pipeline.tools import validate_model, draw_chart, save_model


if __name__ == "__main__":

    NAME_TO_SAVE = 'DP_ema_100_ep_100_data_' + str(int(time.time()))
    PATH_TO_DATA = "/home/andriisydor/masters_thesis/visuomotor_policy/data/pusht_cchi_v7_replay.zarr"
    PATH_TO_STORAGE = "/home/andriisydor/masters_thesis/visuomotor_policy/checkpoints"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CONFIG = DiffusionPolicyConfig()

    BATCH_SIZE = 64
    EPOCHS = 2

    data_split = PushTImageDataset.default_dataset_split()
    dataset_root = zarr.open(PATH_TO_DATA, 'r')
    stats = PushTImageDataset.calculate_train_stats(dataset_root, data_split["train"])

    train_dataset = PushTImageDataset(
        dataset_root=dataset_root,
        split_indexes=data_split["train"],
        pred_horizon=CONFIG.pred_horizon,
        obs_horizon=CONFIG.obs_horizon,
        action_horizon=CONFIG.action_horizon,
        stats=stats)
        
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )

    validation_dataset = PushTImageDataset(
        dataset_root=dataset_root,
        split_indexes=data_split["valid"],
        pred_horizon=CONFIG.pred_horizon,
        obs_horizon=CONFIG.obs_horizon,
        action_horizon=CONFIG.action_horizon,
        stats=stats)
        
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )

    test_dataset = PushTImageDataset(
        dataset_root=dataset_root,
        split_indexes=data_split["test"],
        pred_horizon=CONFIG.pred_horizon,
        obs_horizon=CONFIG.obs_horizon,
        action_horizon=CONFIG.action_horizon,
        stats=stats)
        
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )

    policy = DiffusionPolicy2(CONFIG, DEVICE)

    ema = EMAModel(
        parameters=policy.nets.parameters(),
        power=0.75)

    optimizer = torch.optim.AdamW(
        params=policy.nets.parameters(),
        lr=1e-4, weight_decay=1e-6)

    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(train_dataloader) * EPOCHS
    )

    epoch_losses = []
    validation_losses = []
    with tqdm(range(EPOCHS), desc='Epoch') as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            with tqdm(train_dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # data normalized in dataset
                    # device transfer
                    nimage = nbatch['image'][:, :CONFIG.obs_horizon].float().to(DEVICE)
                    npos = nbatch['agent_pos'][:, :CONFIG.obs_horizon].float().to(DEVICE)
                    naction = nbatch['action'].float().to(DEVICE)
                    B = npos.shape[0]

                    t = policy.sample_t(B)
                    noise = torch.randn(naction.shape, device=DEVICE)
                    noisy_actions = policy.forward_process(naction, noise, t)

                    noise_pred = policy.predict_noise(nimage, npos, noisy_actions, t)

                    loss = nn.functional.mse_loss(noise_pred, noise)

                    loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()
                    ema.step(policy.nets.parameters())

                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            mean_loss = np.mean(epoch_loss)
            epoch_losses.append(mean_loss)
            tglobal.set_postfix(loss=np.mean(mean_loss))
            validation_loss = validate_model(policy, validation_dataloader, CONFIG, DEVICE)
            validation_losses.append(validation_loss)
            print(f'epoch {epoch_idx}: train_loss={mean_loss}, validation_loss={validation_loss}')

    # Weights of the EMA model
    # is used for inference
    ema_policy = DiffusionPolicy2.from_ema(CONFIG, DEVICE, ema)
    save_model(ema_policy.nets, PATH_TO_STORAGE, NAME_TO_SAVE)
    print(f'saved as {NAME_TO_SAVE}')
    draw_chart(epoch_losses, validation_losses)  # TODO: save the chart
    print('test noise MSE: ', validate_model(policy, test_dataloader, CONFIG, DEVICE, nn.functional.mse_loss))
    print('test noise MAE: ', validate_model(policy, test_dataloader, CONFIG, DEVICE, nn.functional.l1_loss))
