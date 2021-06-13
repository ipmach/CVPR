import numpy as np


def box_kernel(m):
    """
    Box kernel
    :param m: size kernel
    :return: kernel
    """
    return np.ones((m, m)) * 1 / m


def gaussian_kernel(m, gamma):
    """
    Gaussian kernel
    :param m: size kernel
    :param gamma: gamma value
    :return: kernel
    """
    a = b = int((m - 1) / 2)
    g = np.zeros((m, m))
    for i, s in enumerate(range(-a, a + 1)):
        for j, t in enumerate(range(-b, b + 1)):
            r2 = s ** 2 + t ** 2
            g[i, j] = np.exp(-r2 / (2 * gamma ** 2))
    G = np.sum(g)
    return g / G


def laplacian_kernel(alternative=False):
    """
    Laplacian kernel
    :param alternative: return alternative kernel (default=False)
    :return: kernel
    """
    if alternative:
        return np.array([[1,  1,  1],
                         [1, -8,  1],
                         [1,  1,  1]])
    return np.array([[0,  1,  0],
                     [1, -4,  1],
                     [0,  1,  0]])


def prewitt_operators():
    """
    Prewitt operators
    :return: kernels
    """
    gx = np.array([[-1, -1,  -1],
                   [0,  0,  0],
                   [1,  1,  1]])
    gy = np.array([[-1,  0,  1],
                   [-1,  0,  1],
                   [-1,  0,  1]])
    return gx, gy


def sobel_operators():
    """
    Sobel operators
    :return: kernel
    """
    gx = np.array([[-1, -2,  -1],
                   [0,  0,  0],
                   [1,  2,  1]])
    gy = np.array([[-1,  0,  1],
                   [-2,  0,  2],
                   [-1,  0,  1]])
    return gx, gy


def box_kernel_l(m):
    """
    Box kernel linear separation
    :param m: size kernel
    :return: kernel
    """
    return np.ones(m) / m ** 2, np.ones(m) / m ** 2


def gaussian_kernel_l(m, gamma):
    """
    Gaussian kernel linear separation
    :param m: size kernel
    :param gamma: gamma value
    :return: kernel
    """
    a = int((m - 1) / 2)
    r = np.zeros(m)
    for i, s in enumerate(range(-a, a +1)):
        r[i] = np.exp(- (s**2) / (2* gamma ** 2))
    R = np.sum(r)
    r = r/R
    return r, r.copy()


def prewitt_operators_l():
    """
    Prewitt operators linear separation
    :return: kernels
    """
    return np.array([1, 1, 1]), \
           np.array([-1, 0, 1]), \
           np.array([-1, 0, 1]), \
           np.array([1, 1, 1])


def sobel_operators_l():
    """
    Sobel operators linear separation
    :return: kernel
    """
    return np.array([1, 2, 1]), \
           np.array([-1, 0, 1]), \
           np.array([-1, 0, 1]), \
           np.array([1, 2, 1])
