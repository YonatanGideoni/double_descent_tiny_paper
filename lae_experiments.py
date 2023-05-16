# Varying train set size with a linear autoencoder
import numpy as np
# Params:
# data_dim = 20 -> 2.4M

# 500k

import torch
from torch.utils.data import TensorDataset, DataLoader

from gen_data import data_gen
from linear_autoencoder import LAE
import wandb

# TODO: Specify these
dataset_sizes = np.linspace(
    start=3,
    stop=21,
    num=40
)
dataset_sizes = np.power(2, dataset_sizes)
dataset_sizes = np.int32(dataset_sizes)
print("Dataset sizes", dataset_sizes)

data_dim = 25
data_latent_dim = 10
batch_size = 20
device = 'cpu'
epochs = 1000
hidden_dim = 100
latent_dim = 20
test_size = 1000
lr = 1e-3

_model = LAE(
    input_dim=data_dim,
    hidden_dim=hidden_dim,
    latent_dim=latent_dim,
    n_hidden_layers=1,
)
params = sum(p.numel() for p in _model.parameters())
print("Params", params)
param_ratio = dataset_sizes / params
print("Param ratio", param_ratio)

for dataset_size in dataset_sizes:

    wandb.init(
        entity='da-some-phone-sunny-emoji',
        project='linear-autoencoder-experiments',
        name=f'lin-ae-dsize-{dataset_size}',
        config={
            "learning_rate": lr,
            "epochs": epochs,
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "batch_size": batch_size,
            "data_dim": data_dim,
            "data_latent_dim": data_latent_dim,
            "test_size": test_size,
            "dataset_size": dataset_size,
            "dataset_sizes": dataset_sizes,
            "n_params": params,
        }
    )

    # Construct dataset
    data = torch.tensor(
        data_gen(
            n_samples=dataset_size + test_size,
            out_dim=data_dim,
            latent_dim=data_latent_dim
        ),
        dtype=torch.float
    ).to(device)
    n_train_samples = dataset_size
    train, test = data[: n_train_samples], data[n_train_samples:n_train_samples + test_size]
    train_set, test_set = TensorDataset(train), TensorDataset(test)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = LAE(
        input_dim=data_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        n_hidden_layers=1,
    )
    print(f'# of params:{sum(p.numel() for p in model.parameters())}')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):

        mean_train_loss = 0
        for x in train_loader:
            optimizer.zero_grad()
            x = x[0]
            x_hat = model(x)
            loss = torch.nn.functional.mse_loss(
                x,
                x_hat
            )
            mean_train_loss += loss
            loss.backward()
            optimizer.step()

        mean_train_loss /= len(train)

        with torch.no_grad():

            mean_test_loss = 0
            for x in test_loader:
                x = x[0]
                x_hat = model(x)
                test_loss = torch.nn.functional.mse_loss(
                    x,
                    x_hat
                )
                mean_test_loss += test_loss.item()
            mean_test_loss /= len(test)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}\t Train loss {mean_train_loss}\t Test loss {mean_test_loss}")

        wandb.log({
            'train_loss': mean_train_loss, 'test_loss': mean_test_loss
        })

    wandb.finish()
