#!/usr/bin/env python3
"""A numpy implementation for "A Bayesian
Variational Principle for Dynamic Self Organized Maps"
"""
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torchvision
import torchvision.transforms as T
from mpl_toolkits.axes_grid1 import ImageGrid
import pickle


class VDSOM():
    def __init__(self, grid_shape, x_dim, checkpoint=None, tore=True,
                 w="normal", sigma=5., eta=1.,
                 adam_a=10**-3, adam_b1=.9, adam_b2=.999, adam_e=1e-8):
        # SOM parameters
        self.grid_shape = grid_shape
        self.n = reduce(lambda a, b: a*b, grid_shape)
        self.x_dim = x_dim
        self.tore = tore
        self.w = np.random.randn(self.n, self.x_dim) if w == "normal" else w
        self.sigma = sigma
        self.eta = eta

        # Initializations
        z_dim = len(grid_shape)
        z = np.stack(np.meshgrid(  # grid points
            *(np.linspace(-1, 1, shape) for shape in grid_shape)),
                     axis=z_dim).reshape((-1, z_dim))
        d = np.abs(z[:, None, :] - z[None, :, :])
        d = np.minimum(d, 2. - d) if tore else d
        self.grid_aff = np.sum(d**2, axis=2)

        # Adam parameters
        self.grad_dim = self.n*self.x_dim + 1
        self.grad = np.empty(self.grad_dim)
        self.adam_m = np.zeros(self.grad_dim)
        self.adam_v = np.zeros(self.grad_dim)
        self.adam_a = adam_a
        self.adam_b1 = adam_b1
        self.adam_b2 = adam_b2
        self.adam_e = adam_e

    def fit(self, X, max_ite=None):
        max_ite = X.shape[0] if max_ite is None else max_ite
        t, n_samples = 0, X.shape[0]
        for t in range(max_ite):
            x = X[t % n_samples].flatten()

            # Stochastic Gradient
            obs_aff = np.sum((x[None, :] - self.w)**2, axis=1)
            lp = self.x_dim*np.log(self.sigma) + .5*self.sigma**-2*obs_aff
            c = self.grid_aff[:, np.argmin(obs_aff)]
            lq = .5*(self.eta*self.sigma)**-2*c
            lq += np.log(np.sum(np.exp(-lq)))
            q = np.exp(-lq)
            d_star = np.sum(c*q, axis=0)
            grad_w = -self.sigma**-2*q[:, None]*(x[None, :] - self.w)
            grad_sigma = (self.eta*(1. + lp - lq)*(c - d_star) - obs_aff)*q
            grad_sigma = self.sigma**-1\
                * (self.x_dim + self.sigma**-2*np.sum(grad_sigma))

            # Adam
            self.grad[:-1] = grad_w.flatten()
            self.grad[-1] = grad_sigma
            self.adam_m = self.adam_b1*self.adam_m +\
                (1 - self.adam_b1)*self.grad
            self.adam_v = self.adam_b2*self.adam_v +\
                (1 - self.adam_b2)*self.grad**2
            m_ = self.adam_m*(1 - self.adam_b1**(t + 1))**-1
            v_ = self.adam_v*(1 - self.adam_b2**(t + 1))**-1
            u = -self.adam_a*m_*(v_**.5 + self.adam_e)**-1
            self.w += u[:-1].reshape(self.w.shape)
            self.sigma += u[-1]


if __name__ == "__main__":
    # Parameters
    grid_shape = (10, 10)
    n_steps = 6000
    overwrite = 1

    # Data
    data = torchvision.datasets.FashionMNIST(
        root="/home/anthony/workspace/data",
        train=True,
        download=True,
        transform=T.ToTensor())
    y = np.stack([d[1] for d in data])
    X = np.stack([d[0] for d in data])

    # Training
    if overwrite:
        x_dim = reduce(lambda a, b: a*b, X[0].shape)
        z_dim = len(grid_shape)
        W = []
        som = VDSOM(grid_shape=grid_shape,
                    x_dim=x_dim, sigma=5.,
                    tore=True, eta=1., adam_a=1e-3)
        labels = np.unique(y)
        for i in labels:
            som.fit(X[y == i])
            W.append(np.copy(som.w))
            print(f"{100*i//len(labels)}%, "
                  f"w={np.mean(som.w):.2e}, "
                  f"sigma={som.sigma:.2e}")
        with open("weigths.pkl", "wb") as f:
            pickle.dump(W, f)
    else:
        with open("weigths.pkl", "rb") as f:
            W = pickle.load(f)

    # Animation
    data_shape = X.shape[1:]
    picture_flag = (len(data_shape) != 1)
    fig = plt.figure(figsize=(16, .8*16))
    lines = {}
    grid = ImageGrid(fig, 111, nrows_ncols=grid_shape)
    im0 = np.transpose(np.ones(data_shape), (1, 2, 0))
    vmin, vmax = np.min(W[-1]), np.max(W[-1])
    lines["w"] = []
    for iax, ax in enumerate(grid):
        ax.set_xticks([])
        ax.set_yticks([])
        line = ax.imshow(im0, animated=True, interpolation="none",
                         vmin=vmin, vmax=vmax)
        lines["w"].append(line)

    def update(frame):
        w = W[frame]
        for line, im in zip(lines["w"], w):
            im = np.transpose(im.reshape(data_shape), (1, 2, 0))
            line.set_array(im)
        return lines["w"]

    anim = FuncAnimation(fig, update, frames=len(W), repeat=True,
                         interval=1000./10, blit=True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    anim.save('VDSOM.gif', writer='imagemagick', fps=1)
