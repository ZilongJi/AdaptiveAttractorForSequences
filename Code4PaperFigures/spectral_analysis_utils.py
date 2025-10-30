from typing import Optional, Callable

import numpy as np


def constructTransmat(
    H: int, 
    W: int, 
    var: Optional[float] = None, 
    offset: int = None
):
    transmat = np.zeros((H*W, H*W))
    for i in range(H):
        for j in range(W):
            a = gaussianPlaceField(i, j, H, W, offset, var=var)
            a = a/np.sum(a)
            transmat[i*W+j, :] = a
    return transmat


def gaussianDistanceKernel(dist: float, var: float):
    return np.exp(-dist/(2*var))/np.sqrt(2*np.pi*var)


def gaussianPlaceField(
    x: int, 
    y: int, 
    H: int, 
    W: int, 
    offset: Optional[int] = None, 
    density: Callable = gaussianDistanceKernel, 
    var: Optional[float] = None
):
    if offset is None:
        offset = np.array([0., 0.])
    if var is None:
        var = np.min([H, W])/10
    m_grid = np.meshgrid(np.arange(W), np.arange(H))
    mean = np.array([y, x]) + offset[[1, 0]]
    dists_xy = np.abs(np.stack([m_grid[0].ravel(), m_grid[1].ravel()], axis=-1) - mean)
    dists_x = np.minimum(dists_xy[:, 0], W - dists_xy[:, 0])
    dists_y = np.minimum(dists_xy[:, 1], H - dists_xy[:, 1])
    dists_xy = np.stack([dists_x, dists_y], axis=-1)
    dists_squared = np.sum(np.square(dists_xy), axis=-1)
    density_matrix_flatten = density(dists_squared, var)
    return density_matrix_flatten


def circulant_perturbation(N: int, d: int, eps: float):
    P = np.zeros((N, N))
    P[np.arange(N-d), np.arange(N-d) + d] = eps
    P[-(np.arange(d)+1), np.arange(d)[::-1]] = eps
    return P


def discrete_fourier_transform(x: np.ndarray):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    X = np.dot(M, x)
    return X


def circulantEvec(N: int, k: int):
    return np.array([np.exp(2*np.pi*complex(0, 1)*j*k/N) for j in range(N)]) / np.sqrt(N)


def circulant_perturbation(N: int, d: int, eps: float):
    P = np.zeros((N, N))
    P[np.arange(N-d), np.arange(N-d) + d] = eps
    P[-(np.arange(d)+1), np.arange(d)[::-1]] = eps
    return P


def theoretical_circulant_perturbation(N: int, d: int, eps: float):
    return eps * np.exp(-1j * 2 * np.pi * np.arange(N) * d / N)


def construct_lazy_randomM_walk_transmat(N: int):
    A = np.zeros((N, N))
    A[np.arange(N), np.arange(N)] = 0.5
    A[np.arange(N-1), np.arange(N-1)+1] = 0.25
    A[np.arange(1, N), np.arange(1, N)-1] = 0.25 
    A[0, -1] = 0.25
    A[-1, 0] = 0.25
    
    return A


def fourierMat(N: int):
    return np.array([[np.exp(2*np.pi*complex(0, 1)/N*j*k) for j in range(N)] for k in range(N)])


def getDFT_evals(T: np.ndarray):
    num_state = T.shape[0]
    fourier_mat = fourierMat(num_state)
    dft = fourier_mat @ T[0, :]
    return dft