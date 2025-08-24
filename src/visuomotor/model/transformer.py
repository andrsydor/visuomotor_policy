import torch
from torch import nn

from visuomotor.model.tools import init_transformer_weights


class ActionTransformer(nn.Module):
    def __init__(self, pred_horizon, action_dim, n_cond_tokens, emb_dim=512, nlayer=7, nhead=8, dropout=0.1) -> None:
        super().__init__()

        self.agent_pos_emb = nn.Linear(action_dim, emb_dim)

        self.out_pos_emb = nn.Parameter(torch.randn(1, pred_horizon, emb_dim))
        self.cond_pos_emb =  nn.Parameter(torch.randn((1, n_cond_tokens, emb_dim)))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=(4 * emb_dim),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=nlayer
        )

        self.final_layer_norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, action_dim)

        self.apply(init_transformer_weights)
    
    def forward(self, img_emb, agent_pos) -> torch.Tensor:
        agent_pos_emb = self.agent_pos_emb(agent_pos)
        obs_emb = torch.cat([img_emb, agent_pos_emb], dim=1)

        obs_emb = obs_emb + self.cond_pos_emb

        B = img_emb.shape[0]
        out_pos_emb = self.out_pos_emb.expand(B, -1, -1)

        y = self.transformer_decoder(
            tgt=out_pos_emb,
            memory=obs_emb
        )
        y = self.final_layer_norm(y)
        y = self.head(y)
        return y