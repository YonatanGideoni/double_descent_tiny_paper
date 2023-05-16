import os
from copy import copy
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader

import wandb
from autoencoder import AE
from consts import DEVICE


@dataclass
class ImgCacher:
    data: list
    cache_dir: str


@dataclass
class ExpParams:
    depth: int
    hidden_dim: int
    latent_dim: int
    input_dim: int
    dataset_size: int
    batch_size: int
    epochs: int
    lr: float
    ld_hidden_exp: int
    ld_depth_exp: int
    hd_depth_exp: int
    hd_late_exp: int
    n_model_params: int = None
    res_cacher: ImgCacher = None


@dataclass
class ExpSetParams:
    def_depth: int
    def_hidden_dim: int
    def_latent_dim: int
    def_input_dim: int
    dataset_size: int
    exp_hidden_dims: List[int]
    exp_latent_dims: List[int]
    exp_depths: List[int]
    ld_hidden_exp: int
    ld_depth_exp: int
    hd_depth_exp: int
    hd_late_exp: int
    batch_size: int
    epochs: int
    res_cacher: ImgCacher = None
    lr: float = 1e-3


class ExpTypes:
    latent = 'latent'
    depth = 'depth'
    hidden_dim = 'hidden_dim'

    @classmethod
    def assert_valid(cls, experiment_type: str):
        if experiment_type not in [cls.latent, cls.depth, cls.hidden_dim]:
            raise Exception(_INVAL_EXP_TYPE_ERR_MSG)


_INVAL_EXP_TYPE_ERR_MSG = "Experiment type not allowed - please use a type defined in ExpTypes"


def get_loss_func():
    return nn.MSELoss()


def create_architecture(exp_params: ExpParams) -> nn.Module:
    return AE(
        input_dim=exp_params.input_dim,
        hidden_dim=exp_params.hidden_dim,
        latent_dim=exp_params.latent_dim,
        n_hidden_layers=exp_params.depth,
        final_activation=nn.Identity(),
    ).to(DEVICE)


# UK spelling especially for Dulhan <3
def get_optimiser(model: nn.Module, lr: float) -> Adam:
    return Adam(model.parameters(), lr=lr)


def count_model_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_epoch(epoch: int, exp_params: ExpParams, model: nn.Module, loss_func: callable, optimiser: Adam,
              train_loader: DataLoader, test_loader: DataLoader, verbose: bool = True) -> dict:
    epoch_train_loss = 0
    train_count = 0
    for batch_idx, x in enumerate(train_loader):
        if len(x) == 2:
            x, _ = x
        x = x[0]
        x = x.to(DEVICE)
        train_count += len(x)

        optimiser.zero_grad()

        x_hat = model(x)
        loss = loss_func(x, x_hat)

        loss.backward()
        optimiser.step()

        epoch_train_loss += loss.item()

    # After each epoch track training and test loss w.r.t. BCE
    with torch.no_grad():
        epoch_test_loss = 0
        test_count = 0
        for batch_idx, x in enumerate(test_loader):
            if len(x) == 2:
                x, _ = x
            x = x[0]
            x = x.to(DEVICE)
            test_count += len(x)
            x_hat = model(x)
            test_loss = loss_func(x, x_hat)
            epoch_test_loss += test_loss.item()

        epoch_train_loss /= train_count
        epoch_test_loss /= test_count

        if verbose:
            print("\tEpoch", epoch + 1, "complete!", "\tAverage Train Loss: ",
                  epoch_train_loss, "\tAverage Test Loss: ", epoch_test_loss)

    return {"train_loss": epoch_train_loss,
            "test_loss": epoch_test_loss,
            "n_model_params": exp_params.n_model_params,
            "dataset_size": exp_params.dataset_size,
            "epoch": epoch
            }


def start_latent_experiment(exps_params: ExpSetParams, generic_exp_params: ExpParams, train_loader: DataLoader,
                            test_loader: DataLoader, mod_prefix: str) -> pd.DataFrame:
    exps_data = []
    for latent_dim in exps_params.exp_latent_dims:
        exp_params = copy(generic_exp_params)
        exp_params.latent_dim = latent_dim
        exp_name = mod_prefix + f'_latent_{latent_dim}'

        # Override hidden dimension
        exp_params.hidden_dim = exp_params.hd_late_exp

        exp_data = run_experiment(exp_params, ExpTypes.latent, exp_name, train_loader, test_loader)
        exps_data.append(exp_data)

    return pd.concat(exps_data) if exps_data else pd.DataFrame()


def start_hidden_experiment(exps_params: ExpSetParams, generic_exp_params: ExpParams, train_loader: DataLoader,
                            test_loader: DataLoader, mod_prefix: str) -> pd.DataFrame:
    exps_data = []
    for hidden_dim in exps_params.exp_hidden_dims:
        exp_params = copy(generic_exp_params)
        exp_params.hidden_dim = hidden_dim
        exp_name = mod_prefix + f'_hidden_{hidden_dim}'

        # Override latent dimension
        exp_params.latent_dim = exp_params.ld_hidden_exp

        exp_data = run_experiment(exp_params, ExpTypes.hidden_dim, exp_name, train_loader, test_loader)
        exps_data.append(exp_data)

    return pd.concat(exps_data) if exps_data else pd.DataFrame()


def start_depth_experiment(exps_params: ExpSetParams, generic_exp_params: ExpParams, train_loader: DataLoader,
                           test_loader: DataLoader, mod_prefix: str) -> pd.DataFrame:
    exps_data = []
    for depth in exps_params.exp_depths:
        exp_params = copy(generic_exp_params)
        exp_params.depth = depth
        exp_name = mod_prefix + f'_depth_{depth}'

        # Override hidden and latent dimension
        exp_params.latent_dim = exp_params.ld_depth_exp
        exp_params.hidden_dim = exp_params.hd_depth_exp

        exp_data = run_experiment(exp_params, ExpTypes.depth, exp_name, train_loader, test_loader)
        exps_data.append(exp_data)

    return pd.concat(exps_data) if exps_data else pd.DataFrame()


def cache_img(img, cache_dir: str, img_name: str):
    plt.imshow(img)

    cache_path = os.path.join(cache_dir, img_name)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    plt.savefig(cache_path)
    plt.close()


def cache_latent_interp_results(orig_img: torch.Tensor, model: AE, img_path: str, n_samples: int = 10,
                                min_mean: float = -3, max_mean: float = 3):
    orig_recon_img, mean, log_var = model(orig_img)
    shape = int(orig_img.shape[1] ** 0.5)
    shape = shape, shape
    orig_recon_img = orig_recon_img.detach().cpu().numpy().reshape(*shape)
    cache_img(orig_recon_img, img_path, 'recon.png')

    log_zero_var = -abs(log_var) * torch.inf
    for latent_var_i in range(mean.shape[1]):
        for latent_val in np.linspace(min_mean, max_mean, n_samples):
            new_mean = mean.numpy()
            new_mean[0, latent_var_i] = latent_val

            recon_img = model.decode(mean, log_zero_var)[0].detach().cpu().numpy().reshape(*shape)
            cache_img(recon_img, img_path, f'{latent_var_i}_{latent_val:.2f}.png')


def cache_results(model: AE, res_cacher: ImgCacher) -> None:
    if res_cacher is None:
        return

    with torch.no_grad():
        for img_num, orig_img in enumerate(res_cacher.data):
            img_path = os.path.join(res_cacher.cache_dir, str(img_num))
            cache_img(orig_img, img_path, 'orig.png')

            orig_img_tensor = torch.Tensor(orig_img).to(DEVICE).view(1, orig_img.shape[0] ** 2) \
                .type('torch.FloatTensor')
            cache_latent_interp_results(orig_img_tensor, model, img_path)


def run_experiment(exp_params: ExpParams, exp_type: str, name: str, train_loader: DataLoader,
                   test_loader: DataLoader, verbose: bool = True) -> pd.DataFrame:
    model = create_architecture(exp_params).to(DEVICE)
    exp_params.n_model_params = count_model_params(model)
    optimiser = get_optimiser(model, exp_params.lr)

    loss_func = get_loss_func()

    config_latent_dim = exp_params.latent_dim
    config_hidden_dim = exp_params.hidden_dim
    if "hidden" in exp_type:
        config_latent_dim = exp_params.ld_hidden_exp
    if "depth" in exp_type:
        config_latent_dim = exp_params.ld_depth_exp
        config_hidden_dim = exp_params.hd_depth_exp
    if "latent" in exp_type:
        config_hidden_dim = exp_params.hd_late_exp

    wandb.init(
        project="day-double-d-tinypaper",
        name=name,
        entity="da-some-phone-sunny-emoji",
        config={
            "learning_rate": exp_params.lr,
            "epochs": exp_params.epochs,
            "latent_dim": config_latent_dim,
            "hidden_dim": config_hidden_dim,
            "batch_size": exp_params.batch_size
        })

    # Training loop
    if verbose:
        print("Start training VAE...")

    model.train()

    run_data = []
    best_train_so_far = np.inf
    best_test_so_far = np.inf
    for epoch in range(exp_params.epochs):
        epoch_data = run_epoch(epoch, exp_params, model, loss_func, optimiser, train_loader, test_loader, verbose)

        best_test_so_far = min(best_test_so_far, epoch_data['test_loss'])
        best_train_so_far = min(best_train_so_far, epoch_data['train_loss'])
        epoch_data['min_test_loss'] = best_test_so_far
        epoch_data['min_train_loss'] = best_train_so_far

        wandb.log(epoch_data)
        run_data.append(epoch_data)

    # Mark one run as finished
    wandb.finish()

    run_data = pd.DataFrame.from_records(run_data)
    run_data['name'] = name
    return run_data


def run_experiments(exps_params: ExpSetParams, name_prefix: str, train_loader: DataLoader,
                    test_loader: DataLoader) -> pd.DataFrame:
    generic_exp_params = ExpParams(depth=exps_params.def_depth, hidden_dim=exps_params.def_hidden_dim,
                                   latent_dim=exps_params.def_latent_dim, input_dim=exps_params.def_input_dim,
                                   dataset_size=exps_params.dataset_size, batch_size=exps_params.batch_size,
                                   epochs=exps_params.epochs, lr=exps_params.lr,
                                   ld_hidden_exp=exps_params.ld_hidden_exp, ld_depth_exp=exps_params.ld_depth_exp,
                                   hd_depth_exp=exps_params.hd_depth_exp, hd_late_exp=exps_params.hd_late_exp)

    all_runs_data = []

    mod_prefix = name_prefix

    # Running latent dimensions experiment
    latent_exp_result = start_latent_experiment(exps_params, generic_exp_params, train_loader, test_loader, mod_prefix)
    all_runs_data.append(latent_exp_result)

    # Running hidden dimensions experiment
    hidden_exp_result = start_hidden_experiment(exps_params, generic_exp_params, train_loader, test_loader, mod_prefix)
    all_runs_data.append(hidden_exp_result)

    # Running depth dimensions experiment
    depth_exp_result = start_depth_experiment(exps_params, generic_exp_params, train_loader, test_loader, mod_prefix)
    all_runs_data.append(depth_exp_result)

    all_runs_data = [data for data in all_runs_data if len(data)]
    return pd.concat(all_runs_data).reset_index(drop=True)
