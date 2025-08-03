import os
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import zarr

from visuomotor.config.base_policy_config import BasePolicyConfig
from visuomotor.task.components import task_from_string


def draw_chart(train_losses, val_losses):
  fig, ax = plt.subplots()

  ax.plot(train_losses, label='train loss', color='maroon')
  ax.plot(val_losses, label='validation loss', color='green')

  ax.set_xlabel('epoch')
  ax.set_ylabel('l2_loss')

  ax.legend()

  plt.show()


def save_model(module, path_to_storage, name):
  torch.save(module.state_dict(), os.path.join(path_to_storage, name + '.ckpt'))


def validate_model(policy, dataloader, config: BasePolicyConfig, device, function=nn.functional.mse_loss):
  policy.nets.eval()
  losses = []
  with torch.no_grad():
    for nbatch in dataloader:
      # data normalized in dataset
      # device transfer
      nimage = nbatch['image'][:, :config.obs_horizon].float().to(device)
      npos = nbatch['agent_pos'][:, :config.obs_horizon].float().to(device)
      naction = nbatch['action'].float().to(device)
      B = npos.shape[0]

      t = policy.sample_t(B)
      noise = torch.randn(naction.shape, device=device)
      noisy_actions = policy.forward_process(naction, noise, t)
      noise_pred = policy.predict_noise(nimage, npos, noisy_actions, t)

      loss = function(noise_pred, noise)
      loss_cpu = loss.item()
      losses.append(loss_cpu)
  mean_loss = np.mean(losses)
  policy.nets.train()
  return mean_loss


def calculate_error(policy, dataloader, config: BasePolicyConfig, device, function=nn.functional.mse_loss):
  policy.nets.eval()
  losses = []
  with torch.no_grad():
    for nbatch in dataloader:
      # data normalized in dataset
      # device transfer
      nimage = nbatch['image'][:, :config.obs_horizon].float().to(device)
      npos = nbatch['agent_pos'][:, :config.obs_horizon].float().to(device)
      naction = nbatch['action'].float().to(device)
      B = npos.shape[0]

      pred_naction = policy.action(nimage, npos, B)

      loss = function(pred_naction, naction)
      loss_cpu = loss.item()
      losses.append(loss_cpu)
  mean_loss = np.mean(losses)
  policy.nets.train()
  return mean_loss


def create_dataloaders(
    task_name: str,
    path_to_data: str,
    batch_size: int,
    pred_horizon: int,
    obs_horizon: int,
    action_horizon: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    # train, val, test

    task = task_from_string(task_name)
    dataset_class = task.dataset_class()

    data_split = dataset_class.default_dataset_split()
    dataset_root = zarr.open(path_to_data, 'r')
    stats = dataset_class.calculate_train_stats(dataset_root, data_split["train"])

    train_dataset = dataset_class(
        dataset_root=dataset_root,
        split_indexes=data_split["train"],
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        stats=stats)
        
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )

    validation_dataset = dataset_class(
        dataset_root=dataset_root,
        split_indexes=data_split["valid"],
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        stats=stats)
        
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )

    test_dataset = dataset_class(
        dataset_root=dataset_root,
        split_indexes=data_split["test"],
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        stats=stats)
        
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )

    return train_dataloader, validation_dataloader, test_dataloader
