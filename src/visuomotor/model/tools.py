import torch
from torch import nn


def init_transformer_weights(module: nn.Module) -> None:
    def w_init_func(w):
        torch.nn.init.normal_(w, mean=0.0, std=0.02)
    def b_init_func(b):
        torch.nn.init.zeros_(b)

    if isinstance(module, (nn.Linear)):
        w_init_func(module.weight)
        if module.bias is not None:
            b_init_func(module.bias)
    
    elif isinstance(module, nn.MultiheadAttention):
        weight_names = ['in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
        for name in weight_names:
            weight = getattr(module, name)
            if weight is not None:
                w_init_func(weight)
        
        bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
        for name in bias_names:
            bias = getattr(module, name)
            if bias is not None:
                b_init_func(bias)
    elif isinstance(module, nn.LayerNorm):
        b_init_func(module.bias)
        torch.nn.init.ones_(module.weight)


def get_optim_groups(self, weight_decay: float=1e-3):
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in self.named_modules():
        for pn, p in m.named_parameters():
            fpn = "%s.%s" % (mn, pn) if mn else pn

            if pn.endswith("bias"):
                no_decay.add(fpn)
            elif pn.startswith("bias"):
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)

    no_decay.add("out_pos_emb")
    if self.cond_pos_emb is not None:
        no_decay.add("cond_pos_emb")

    param_dict = {pn: p for pn, p in self.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert (
        len(inter_params) == 0
    ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert (
        len(param_dict.keys() - union_params) == 0
    ), "parameters %s were not separated into either decay/no_decay set!" % (
        str(param_dict.keys() - union_params),
    )

    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": weight_decay,
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0,
        },
    ]
    return optim_groups


def construct_default_optimizer(policy):
    return torch.optim.AdamW(
        params=policy.nets.parameters(),
        lr=1e-4, weight_decay=1e-6)


def construct_transformer_optimizer(policy):
    groups = get_optim_groups(policy.nets["action_predictor"], weight_decay=1e-6)
    return torch.optim.AdamW(
        params=groups,
        lr=1e-4)
