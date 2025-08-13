
class BasePolicyConfig:
  def __init__(self):
    self.pred_horizon = 16
    self.obs_horizon = 2
    self.action_horizon = 8

    self.action_dim = 2
    self.pos_dim = 2
