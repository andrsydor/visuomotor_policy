import torch
from torch import nn

from visuomotor.model.resnet import get_resnet, replace_bn_with_gn
from visuomotor.model.transformer import ActionTransformer


class ExplicitPolicy:

  def __init__(self, config, device):
    self.config = config
    self.device = device

    assert self.config.image_obs_horizon == 1

    vision_encoder = get_resnet('resnet34')
    vision_encoder = replace_bn_with_gn(vision_encoder)

    action_predictor =  ActionTransformer(config.pred_horizon, config.action_dim, 4)

    self.nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'action_predictor': action_predictor
    })

    _ = self.nets.to(device)

  def action(self, arm_image, depth_image, agent_pos, batch_size):
    arm_image_features = self.nets['vision_encoder'](arm_image.flatten(end_dim=1))
    depth_image_features = self.nets['vision_encoder'](depth_image.flatten(end_dim=1))
    
    obs = torch.cat([arm_image_features.unsqueeze(1), depth_image_features.unsqueeze(1)], dim=1)

    naction = self.nets['action_predictor'](obs, agent_pos)

    return naction
  
  def compute_train_loss(self, arm_image: torch.Tensor, depth_image: torch.Tensor, poses: torch.Tensor, target_actions: torch.Tensor) -> torch.Tensor:
    batch_size = poses.shape[0]

    pred_actions = self.action(arm_image, depth_image, poses, batch_size)
    loss = nn.functional.l1_loss(pred_actions, target_actions)

    return loss

  @staticmethod
  def from_ema(config, device, ema):
    new_policy = ExplicitPolicy(config, device)
    ema.copy_to(new_policy.nets.parameters())
    return new_policy
