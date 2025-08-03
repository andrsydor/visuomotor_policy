import time
import argparse
from typing import Tuple

import torch
from torch import nn
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
import numpy as np
from tqdm.auto import tqdm

from visuomotor.config.diffusion_policy_config import DiffusionPolicyConfig
from visuomotor.policy.diffusion_policy import DiffusionPolicy2
from visuomotor.pipeline.tools import validate_model, draw_chart, save_model, create_dataloaders


POLICIES = {
    "dp": (DiffusionPolicy2, DiffusionPolicyConfig)
}


def choose_policy(policy_name):
    policy_pair = POLICIES.get(policy_name, None)
    if not policy_pair:
        raise NotImplementedError("No such policy")
    return policy_pair


def main():

    parser = argparse.ArgumentParser(description="Dataset demo")
    parser.add_argument("--dataset", help="Path to the zarr file")  # "/home/andriisydor/masters_thesis/visuomotor_policy/data/pusht_cchi_v7_replay.zarr"
    parser.add_argument("--storage", help="Where to store checkpoints")  # "/home/andriisydor/masters_thesis/visuomotor_policy/checkpoints"
    parser.add_argument("--task", help="Task name")  # "push_t"
    parser.add_argument("--name", help="How to name checkpoints")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, help="Batch_size")
    parser.add_argument("--policy", help="Policy name")
    args = parser.parse_args()

    PATH_TO_DATA = args.dataset
    TASK_NAME = args.task
    PATH_TO_STORAGE = args.storage
    NAME_TO_SAVE = args.name + str(int(time.time()))
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    policy_class, policy_config_class = choose_policy(args.policy)
    CONFIG = policy_config_class()
    policy = policy_class(CONFIG, DEVICE)

    train_dataloader, validation_dataloader, test_dataloader = create_dataloaders(
        TASK_NAME,
        PATH_TO_DATA,
        BATCH_SIZE,
        CONFIG.pred_horizon,
        CONFIG.obs_horizon,
        CONFIG.action_horizon
    )

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

    with tqdm(range(EPOCHS), desc='Epoch') as tepochs:
        
        for epoch_i in tepochs:
            epoch_loss = list()
            
            with tqdm(train_dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:

                    # TODO: pass the dictionary
                    images = nbatch['image'].float().to(DEVICE)
                    poses = nbatch['agent_pos'].float().to(DEVICE)
                    target_actions = nbatch['action'].float().to(DEVICE)

                    loss = policy.compute_train_loss(images, poses, target_actions)

                    loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                    ema.step(policy.nets.parameters())

                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)

            mean_loss = np.mean(epoch_loss)
            epoch_losses.append(mean_loss)
            tepochs.set_postfix(loss=np.mean(mean_loss))
            validation_loss = validate_model(policy, validation_dataloader, CONFIG, DEVICE)
            validation_losses.append(validation_loss)
            print(f'epoch {epoch_i}: train_loss={mean_loss}, validation_loss={validation_loss}')

    ema_policy = policy_class.from_ema(CONFIG, DEVICE, ema)
    save_model(ema_policy.nets, PATH_TO_STORAGE, NAME_TO_SAVE)
    print(f'saved as {NAME_TO_SAVE}')
    draw_chart(epoch_losses, validation_losses)  # TODO: save the chart
    print('test noise MSE: ', validate_model(policy, test_dataloader, CONFIG, DEVICE, nn.functional.mse_loss))
    print('test noise MAE: ', validate_model(policy, test_dataloader, CONFIG, DEVICE, nn.functional.l1_loss))


if __name__ == "__main__":
    main()
