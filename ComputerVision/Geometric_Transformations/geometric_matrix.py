import numpy as np


class GeometricTransformations:
    """
    Geometric transformations in images
    """

    @staticmethod
    def identity():
        """
        Create identity transformation
        :return: transformation matrix
        """
        return np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]])

    @staticmethod
    def scaling_k(cx, cy):
        """
        Create scaling transformation
        :param cx: scaling factor x axis
        :param cy: scaling factor y axis
        :return: transformation matrix
        """
        return np.array([[cx, 0, 0],
                         [0, cy, 0],
                         [0, 0, 1]])

    @staticmethod
    def translation_k(tx, ty):
        """
        Create translation transformation
        :param tx: translation in x axis
        :param ty: translation in y axis
        :return: transformation matrix
        """
        return np.array([[1, 0, tx],
                         [0, 1, ty],
                         [0, 0, 1]])

    @staticmethod
    def rotation_k(angle):
        """
        Create rotation transformation
        :param angle: angle rotation
        :return: transformation matrix
        """
        return np.array([[np.cos(angle), -1 * np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0],
                         [0, 0, 1]])

    @staticmethod
    def vertical_shear_k(lambda_):
        """
        Create vertical shear transformation
        :param lambda_: factor vertical shearing
        :return: transformation matrix
        """
        return np.array([[1, lambda_, 0],
                         [0, 1, 0],
                         [0, 0, 1]])

    @staticmethod
    def horizontal_shear_k(lambda_):
        """
        Create horizontal sher transformation
        :param lambda_: factor horizontal shearing
        :return: transformation matrix
        """
        return np.array([[1, 0, 0],
                         [lambda_, 1, 0],
                         [0, 0, 1]])
