import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from experiment_utils import ExpSetParams, run_experiments
from gen_data import data_gen

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 10
data_dim = 50
epochs = 200
train_ds_size = 5000
test_ds_size = 10000
lr = 1e-3

for hidden_dim in list(range(160, 200, 8)):
    # Generate dataset
    train_data = torch.tensor(data_gen(train_ds_size, data_dim, 20), dtype=torch.float).to(device)
    test_data = torch.tensor(data_gen(test_ds_size, data_dim, 20), dtype=torch.float).to(device)
    train_set, test_set = TensorDataset(train_data), TensorDataset(test_data)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    exps_params = ExpSetParams(def_depth=0, def_hidden_dim=hidden_dim, def_latent_dim=24, def_input_dim=data_dim,
                               dataset_size=len(train_loader),
                               exp_hidden_dims=[],
                               exp_depths=[],
                               exp_latent_dims=[1500, 2000, 2500, 3000, 4000, 5000, 7500, 10_000],
                               batch_size=batch_size, epochs=epochs,
                               ld_hidden_exp=32, ld_depth_exp=16, hd_depth_exp=24, hd_late_exp=hidden_dim, lr=lr)

    # Run experiments
    experiments_res = run_experiments(exps_params, f'AE-run-inp{data_dim}-ds{train_ds_size}-hid{hidden_dim}',
                                      train_loader, test_loader)
