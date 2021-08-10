from ComputerVision.Geometric_Transformations.geometric_matrix import GeometricTransformations
from ComputerVision.Geometric_Transformations.interpolations import Interpolations
from tqdm.notebook import tqdm
import numpy as np


def forward_mapping(img, A, default_value=255, scale=(1, 1), apply_in_zero=False):
    """
    Apply forward mapping in a image
    :param img: original image
    :param A: matrix transformation
    :param default_value: default value of pixel
    :param scale: scale factor (x,y)
    :param apply_in_zero: apply translation to 0 and back, default=False
    :return: new img
    """
    # New image
    (M, N) = img.shape[:2]
    M2 = int(M * scale[0])
    N2 = int(N * scale[1])
    img2 = np.ones((M2, N2)) * default_value
    if apply_in_zero:
        A_t = GeometricTransformations.translation_k(np.round(M) / 2, np.round(N) / 2)
        A_t_inv = GeometricTransformations.translation_k(-np.round(M) / 2, -np.round(N) / 2)
    # Apply mapping
    for k in tqdm(range(M)):
        for l in range(N):
            x = k + 0.5
            y = l + 0.5
            p = np.array([x, y, 1])
            if apply_in_zero:
                p = np.matmul(A_t_inv, p)
            q = np.matmul(A, p)
            if apply_in_zero:
                q = np.matmul(A_t, q)
            [x, y, _] = q
            i = round(x - 0.5)
            j = round(y - 0.5)
            try:
                img2[i, j] = img[k, l]
            except IndexError:
                pass  # Out of index

    return img2


def inverse_mapping(img, A, default_value=255,
                    interpolation=Interpolations.bilinear, scale=(1, 1),
                    apply_in_zero=False):
    """
    Apply inverse mapping in a image
    :param img: original image
    :param A: matrix transformation
    :param default_value: default value of pixel
    :param interpolation: interpolation apply
    :param scale: scale factor (x,y)
    :param apply_in_zero: apply translation to 0 and back, default=False
    :return: new img
    """
    # Invert A
    A = np.linalg.inv(A)

    # New image
    (M, N) = img.shape[:2]
    M2 = int(M * scale[0])
    N2 = int(N * scale[1])
    img2 = np.ones((M2, N2)) * default_value
    if apply_in_zero:
        A_t = GeometricTransformations.translation_k(np.round(M) / 2, np.round(N) / 2)
        A_t_inv = GeometricTransformations.translation_k(-np.round(M) / 2, -np.round(N) / 2)
    # Apply mapping
    for i in tqdm(range(M2)):
        for j in range(N2):
            x = i + 0.5
            y = j + 0.5
            p = np.array([x, y, 1])
            if apply_in_zero:
                p = np.matmul(A_t_inv, p)
            q = np.matmul(A, p)
            if apply_in_zero:
                q = np.matmul(A_t, q)
            [x, y, _] = q
            # Interpolation
            img2[i, j] = interpolation(img, M, N, x, y)
    img2 = np.vectorize(lambda x: default_value if np.isnan(x) else x)(img2)
    return img2
