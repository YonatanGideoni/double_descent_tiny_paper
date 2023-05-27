# Code for "No Double Descent in Self-Supervised Learning"

For the AE experiments --- the results used to construct the main results in the paper are given in `results/exp_res_final.csv`. These results are downloaded from `wandb`, hence the format. You can get them by running `ae_experiments.py` with the configurations given there. You can recreate the plots by running `plot_heatmap.ipynb`. 

For the LAE experiments --- the results used to construct the main results in the paper are given in `results/lae_exps_res.csv`. These results are downloaded from `wandb`, hence the format. You can get them by running `lae_experiments.py` with the configurations given there. You can recreate the plots by running `plot_lae_res.ipynb`.

When citing this in academic work please use the following:
```bibtex
@misc{
  jayalath2023no,
  title={No Double Descent in Self-Supervised Learning},
  author={Dulhan Hansaja Jayalath and Alisia Maria Lupidi and Yonatan Gideoni},
  year={2023},
  url={https://openreview.net/forum?id=qNJRvdKDGYg}
}
```