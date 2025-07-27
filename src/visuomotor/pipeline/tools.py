import os

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from visuomotor.config.base_policy_config import BasePolicyConfig


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


