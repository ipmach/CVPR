import numpy as np


class Intensities:

    @staticmethod
    def filter_transform(r, r0=0, r2=1, binary=False):
        if r0 <= r <= r2:
            return 1. if binary else r
        else:
            return 0.

    @staticmethod
    def filter_transform_v(r, r0=0, r2=1, binary=False):
        f = np.vectorize(Intensities.filter_transform)
        return f(r, r0=r0, r2=r2, binary=binary)

    @staticmethod
    def inv_transform(r, r0=0, r2=1):
        if r0 <= r <= r2:
            return 1 - r
        else:
            return r

    @staticmethod
    def inv_transform_v(r, r0=0, r2=1):
        f = np.vectorize(Intensities.inv_transform)
        return f(r, r0=r0, r2=r2)

    @staticmethod
    def log_transform(r, alpha, r0=0, r2=1):
        beta = 1 / (np.exp(1 / alpha) - 1)
        if r0 <= r <= r2:
            return alpha * np.log(r / beta + 1)
        else:
            return r

    @staticmethod
    def log_transform_v(r, alpha, r0=0, r2=1):
        f = np.vectorize(Intensities.log_transform)
        return f(r, alpha, r0=r0, r2=r2)

    @staticmethod
    def exp_transform(r, alpha, r0=0, r2=1):
        beta = 1 / (np.exp(1 / alpha) - 1)
        if r0 <= r <= r2:
            return beta * (np.exp(r / alpha) - 1)
        else:
            return r

    @staticmethod
    def exp_transform_v(r, alpha, r0=0, r2=1):
        f = np.vectorize(Intensities.exp_transform)
        return f(r, alpha, r0=r0, r2=r2)

    @staticmethod
    def gamma_correction(r, gamma, r0=0, r2=1):
        if r0 <= r <= r2:
            return np.power(r, gamma)
        else:
            return r

    @staticmethod
    def gamma_correction_v(r, gamma, r0=0, r2=1):
        f = np.vectorize(Intensities.gamma_correction)
        return f(r, gamma, r0=r0, r2=r2)

    @staticmethod
    def constract_stretching(r, gamma, lambda_, r0=0, r2=1):
        if r0 <= r <= r2:
            return 0.5 * np.arctan(lambda_ *
                                   (np.arctan(gamma * (2 * r - 1))
                                    / np.arctan(gamma)) + 1)
        else:
            return r

    @staticmethod
    def constract_stretching_v(r, gamma, lambda_, r0=0, r2=1):
        f = np.vectorize(Intensities.constract_stretching)
        return f(r, gamma, lambda_, r0=r0, r2=r2)
