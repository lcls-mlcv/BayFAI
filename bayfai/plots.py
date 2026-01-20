import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def plot_pairwise_heatmaps(X, scores, param_names, best_params, n_grid=101):
    """
    Pairwise heatmaps of scores.

    Parameters
    ----------
    X : ndarray, shape (N, D)
        Parameter samples (grid search points).
    scores : ndarray, shape (N,)
        Score values for each sample.
    param_names : list of str
        Names of parameters in order (length D).
    best_params : ndarray, shape (D,)
        Best parameter set.
    n_grid : int
        Grid resolution for interpolation.
    """
    n_params = X.shape[1]-1 # rot3 is fixed
    fig, axes = plt.subplots(n_params, n_params, figsize=(3*n_params, 3*n_params))

    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            if i <= j:
                ax.axis("off")
                continue

            x, y, z = X[:, j], X[:, i], scores

            xi = np.linspace(x.min(), x.max(), n_grid)
            yi = np.linspace(y.min(), y.max(), n_grid)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi = griddata((x, y), z, (Xi, Yi), method="linear")

            ax.scatter(best_params[j], best_params[i], color="red", marker="*", s=120, edgecolor="k")

            im = ax.pcolormesh(Xi, Yi, Zi, shading="auto", cmap="viridis")
            
            fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.05)

            if j == 0:
                ax.set_ylabel(param_names[i])
            if i == n_params - 1:
                ax.set_xlabel(param_names[j])

    fig.tight_layout()
    return fig
