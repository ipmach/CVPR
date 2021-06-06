import numpy as np


class GeometricTransformations:

    @staticmethod
    def identity():
        return np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]])

    @staticmethod
    def scaling_k(cx, cy):
        return np.array([[cx, 0, 0],
                         [0, cy, 0],
                         [0, 0, 1]])

    @staticmethod
    def translation_k(tx, ty):
        return np.array([[1, 0, tx],
                         [0, 1, ty],
                         [0, 0, 1]])

    @staticmethod
    def rotation_k(angle):
        return np.array([[np.cos(angle), -1 * np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0],
                         [0, 0, 1]])

    @staticmethod
    def vertical_shear_k(lambda_):
        return np.array([[1, lambda_, 0],
                         [0, 1, 0],
                         [0, 0, 1]])

    @staticmethod
    def horizontal_shear_k(lambda_):
        return np.array([[1, 0, 0],
                         [lambda_, 1, 0],
                         [0, 0, 1]])
