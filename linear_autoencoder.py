# based on https://github.com/mrdvince/autoencoders/blob/master/linear.py

import torch.nn.functional as F
import torch.nn as nn
import torch


class LAE(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 latent_dim: int,
                 n_hidden_layers: int):

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_hidden_layers = n_hidden_layers
        # encoder
        self.encoder_input_layer = nn.Linear(
            in_features=input_dim, out_features=hidden_dim
        )
        self.encoder_hidden_layers = nn.ModuleList(
            [nn.Linear(in_features=hidden_dim, out_features=hidden_dim) for i in range(n_hidden_layers)]
        )
        self.encoder_output_layer = nn.Linear(
            in_features=hidden_dim, out_features=latent_dim
        )
        # decoder
        self.decoder_input_layer = nn.Linear(
            in_features=latent_dim, out_features=hidden_dim
        )
        self.decoder_hidden_layers = nn.ModuleList(
            [nn.Linear(in_features=hidden_dim, out_features=hidden_dim) for i in range(n_hidden_layers)]
        )
        self.decoder_output_layer = nn.Linear(
            in_features=hidden_dim, out_features=input_dim
        )

    def forward(self, x):
        x = self.encoder_input_layer(x)
        for i in range(self.n_hidden_layers):
            x = self.encoder_hidden_layers[i](x)
        x = self.encoder_output_layer(x)

        x = self.decoder_input_layer(x)
        for i in range(self.n_hidden_layers):
            x = self.decoder_hidden_layers[i](x)
        x = self.decoder_output_layer(x)
        return x
