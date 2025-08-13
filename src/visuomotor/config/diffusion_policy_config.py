from visuomotor.config.base_policy_config import BasePolicyConfig


class DiffusionPolicyConfig(BasePolicyConfig):
    def __init__(self):
        super().__init__()


class DiffusionPolicy2CamConfig(DiffusionPolicyConfig):
    def __init__(self):
        super().__init__()

        # self.pred_horizon = 16
        # self.obs_horizon = 2
        # self.action_horizon = 8

        self.action_dim = 12
        self.pos_dim = 12

        self.image_obs_horizon = 1
