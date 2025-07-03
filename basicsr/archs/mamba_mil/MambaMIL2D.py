from typing import Optional

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .mamba_simple import MambaConfig as SimpleMambaConfig
from .mamba_simple import Mamba as SimpleMamba


# noinspection PyPep8Naming
class MambaMIL2D(nn.Module):
    def __init__(self,
                 in_dim: int = 1024,
                 d_model: int = 128,
                 n_layers: int = 1,
                 d_state: int = 16,
                 inner_layernorms: bool = False,
                 drop_out: float = 0.25,
                 pscan: bool = True,
                 cuda_pscan: bool = False,
                 mamba_2d_max_w: int = 224,
                 mamba_2d_max_h: int = 224,
                 mamba_2d_pad_token: str = 'trainable',
                 mamba_2d_patch_size: int = 512,
                 n_classes: int = 1,
                 survival: bool = False,
                 pos_emb_type: Optional[str] = None,
                 pos_emb_dropout: float = 0.0,
                 patch_encoder_batch_size: int = 128,
                 ):

        super(MambaMIL2D, self).__init__()

        self.in_dim = in_dim
        self.d_model = d_model
        self.drop_out = drop_out
        self.n_layers = n_layers
        self.d_state = d_state
        self.inner_layernorms = inner_layernorms
        self.pscan = pscan
        self.cuda_pscan = cuda_pscan
        self.mamba_2d_max_w = mamba_2d_max_w
        self.mamba_2d_max_h = mamba_2d_max_h
        self.mamba_2d_pad_token = mamba_2d_pad_token
        self.mamba_2d_patch_size = mamba_2d_patch_size
        self.n_classes = n_classes
        self.survival = survival
        self.pos_emb_type = pos_emb_type
        self.pos_emb_dropout = pos_emb_dropout
        self.patch_encoder_batch_size = patch_encoder_batch_size

        self._fc1 = [nn.Linear(self.in_dim, self.d_model)]
        self._fc1 += [nn.GELU()]
        if self.drop_out > 0:
            self._fc1 += [nn.Dropout(self.drop_out)]

        self._fc1 = nn.Sequential(*self._fc1)

        self.norm = nn.LayerNorm(self.d_model)

        self.layers = nn.ModuleList()
        self.patch_encoder_batch_size = self.patch_encoder_batch_size
        config = SimpleMambaConfig(
            d_model=self.d_model,
            n_layers=self.n_layers,
            d_state=self.d_state,
            inner_layernorms=self.inner_layernorms,
            pscan=self.pscan,
            use_cuda=self.cuda_pscan,
            mamba_2d=True,
            mamba_2d_max_w=self.mamba_2d_max_w,
            mamba_2d_max_h=self.mamba_2d_max_h,
            mamba_2d_pad_token=self.mamba_2d_pad_token,
            mamba_2d_patch_size=self.mamba_2d_patch_size
        )
        self.layers = SimpleMamba(config)
        self.config = config

        self.attention = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.classifier = nn.Linear(self.d_model, self.n_classes)

        if self.pos_emb_type == 'linear':
            self.pos_embs = nn.Linear(2, self.d_model)
            self.norm_pe = nn.LayerNorm(self.d_model)
            self.pos_emb_dropout = nn.Dropout(self.pos_emb_dropout)
        else:
            self.pos_embs = None

        self.apply(_initialize_weights)

    def forward(self, x, coords):
        if len(x.shape) == 2:
            x = x.expand(1, -1, -1)  # (1, num_patch, feature_dim)
        h = x.float()  # [1, num_patch, feature_dim]

        h = self._fc1(h)  # [1, num_patch, mamba_dim];   project from feature_dim -> mamba_dim

        # Add Pos_emb
        if self.pos_emb_type == 'linear':
            pos_embs = self.pos_embs(coords)
            h = h + pos_embs.unsqueeze(0)
            h = self.pos_emb_dropout(h)

        h = self.layers(h, coords, self.pos_embs)

        h = self.norm(h)  # LayerNorm
        A = self.attention(h)  # [1, W, H, 1]

        if len(A.shape) == 3:
            A = torch.transpose(A, 1, 2)
        else:
            A = A.permute(0, 3, 1, 2)
            A = A.view(1, 1, -1)
            h = h.view(1, -1, self.config.d_model)

        A = F.softmax(A, dim=-1)  # [1, 1, num_patch]  # A: attention weights of patches
        h = torch.bmm(A, h)  # [1, 1, 512] , weighted combination to obtain slide feature
        h = h.squeeze(0)  # [1, 512], 512 is the slide dim

        logits = self.classifier(h)  # [1, n_classes]
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        results_dict = None

        if self.survival:
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, None, None  # same return as other models

        return logits, Y_prob, Y_hat, results_dict, None  # same return as other models

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fc1 = self._fc1.to(device)
        self.layers = self.layers.to(device)

        self.attention = self.attention.to(device)
        self.norm = self.norm.to(device)
        self.classifier = self.classifier.to(device)


def _split_tensor(data, batch_size):
    num_chk = int(np.ceil(data.shape[0] / batch_size))
    return torch.chunk(data, num_chk, dim=0)


def _initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
