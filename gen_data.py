import numpy as np


# based on section 3 in https://openreview.net/pdf?id=ieWqvOiKgz2
def data_gen(n_samples: int, out_dim: int, latent_dim: int, rand_mat: np.ndarray = None, snr: float = 10) -> np.ndarray:
    if rand_mat is None:
        rand_mat = np.random.randn(out_dim, latent_dim)
        rand_mat *= snr / latent_dim ** 0.5
    else:
        assert out_dim == rand_mat.shape[0], 'Error - misspecified random matrix, wrong output dim'

    noise = np.random.randn(n_samples, out_dim)
    latent_features = np.random.randn(n_samples, latent_dim)
    return latent_features @ rand_mat.T + noise


if __name__ == '__main__':
    print(data_gen(10 ** 5, 100, 10))
