import torch
from torch import nn

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from visuomotor.model.resnet import get_resnet, replace_bn_with_gn
from visuomotor.model.unet1d import ConditionalUnet1D


class DiffusionPolicy2:

  def __init__(self, config, device):
    self.config = config
    self.device = device
    vision_encoder = get_resnet('resnet18')
    vision_encoder = replace_bn_with_gn(vision_encoder)

    vision_feature_dim = 512
    obs_dim = vision_feature_dim + config.pos_dim

    noise_pred_net = ConditionalUnet1D(
        input_dim=config.action_dim,
        global_cond_dim=(obs_dim * config.obs_horizon)
    )

    self.nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'noise_pred_net': noise_pred_net
    })

    _ = self.nets.to(device)

    self.num_diffusion_iters = 100
    self.noise_scheduler = DDPMScheduler(
        num_train_timesteps=self.num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

  def initial_noise(self, batch_size):
    return torch.randn(
        (batch_size, self.config.pred_horizon, self.config.action_dim),
        device=self.device
        )

  def sample_t(self, batch_size):
    return torch.randint(0, self.noise_scheduler.config.num_train_timesteps,
                         size=(batch_size,),
                         device=self.device
                         ).long()

  def forward_process(self, actions, noise, t):
    return self.noise_scheduler.add_noise(actions, noise, t)

  def predict_noise(self, image, agent_pos, noised_action, t):
    image_features = self.nets['vision_encoder'](image.flatten(end_dim=1))
    image_features = image_features.reshape(*image.shape[:2], -1)
    obs = torch.cat([image_features, agent_pos], dim=-1)
    obs_cond = obs.flatten(start_dim=1)

    noise = self.nets['noise_pred_net'](
        sample=noised_action,
        timestep=t,
        global_cond=obs_cond)

    return noise

  def action(self, image, agent_pos, batch_size):
    naction = self.initial_noise(batch_size)
    self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

    for t in self.noise_scheduler.timesteps:
      noise_pred = self.predict_noise(image, agent_pos, naction, t)
      # remove noise
      naction = self.noise_scheduler.step(
          model_output=noise_pred,
          timestep=t,
          sample=naction
          ).prev_sample
    return naction

  @staticmethod
  def from_ema(config, device, ema):
    new_policy = DiffusionPolicy2(config, device)
    ema.copy_to(new_policy.nets.parameters())
    return new_policy