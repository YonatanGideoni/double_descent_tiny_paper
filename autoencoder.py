import torch.nn as nn


# based on https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1
class AE(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 latent_dim: int,
                 n_hidden_layers: int,
                 final_activation: nn.Module,
                 **kwargs):

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_hidden_layers = n_hidden_layers
        self.final_activation = final_activation

        self.encoder_input_layer = nn.Linear(
            in_features=input_dim, out_features=hidden_dim
        )
        self.encoder_hidden_layers = nn.ModuleList(
            [nn.Linear(in_features=hidden_dim, out_features=hidden_dim) for i in range(n_hidden_layers)]
        )
        self.encoder_output_layer = nn.Linear(
            in_features=hidden_dim, out_features=latent_dim
        )

        self.decoder_input_layer = nn.Linear(
            in_features=latent_dim, out_features=hidden_dim
        )
        self.decoder_hidden_layers = nn.ModuleList(
            [nn.Linear(in_features=hidden_dim, out_features=hidden_dim) for i in range(n_hidden_layers)]
        )
        self.decoder_output_layer = nn.Linear(
            in_features=hidden_dim, out_features=input_dim
        )

        self.activation = nn.ReLU()

    def forward(self, x):

        x = self.activation(self.encoder_input_layer(x))
        for i in range(self.n_hidden_layers):
            x = self.activation(self.encoder_hidden_layers[i](x))
        x = self.activation(self.encoder_output_layer(x))

        x = self.activation(self.decoder_input_layer(x))
        for i in range(self.n_hidden_layers):
            x = self.activation(self.decoder_hidden_layers[i](x))
        x = self.final_activation(self.decoder_output_layer(x))

        y = x

        return y
