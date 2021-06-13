from ComputerVision.Geometric_Transformations.interpolations import Interpolations
from tqdm.notebook import tqdm
import numpy as np


def inverse_mapping(img, A, default_value=255,
                    interpolation=Interpolations.bilinear, scale=(1, 1)):
    """
    Apply inverse mapping in a image
    :param img: original image
    :param A: matrix transformation
    :param default_value: default value of pixel
    :param interpolation: interpolation apply
    :param scale: scale factor (x,y)
    :return: new img
    """
    # Invert A
    A = np.linalg.inv(A)

    # New image
    (M, N) = img.shape[:2]
    M2 = int(M * scale[0])
    N2 = int(N * scale[1])
    img2 = np.ones((M2, N2)) * default_value

    # Apply mapping
    for i in tqdm(range(M2)):
        for j in range(N2):
            x = i + 0.5
            y = j + 0.5
            p = np.array([x, y, 1])
            q = np.matmul(A, p)
            [x, y, _] = q
            # Interpolation
            img2[i, j] = interpolation(img, M, N, x, y)

    return img2


def forward_mapping(img, A, default_value=255, scale=(1, 1)):
    """
    Apply forward mapping in a image
    :param img: original image
    :param A: matrix transformation
    :param default_value: default value of pixel
    :param scale: scale factor (x,y)
    :return: new img
    """
    # New image
    (M, N) = img.shape[:2]
    M2 = int(M * scale[0])
    N2 = int(N * scale[1])
    img2 = np.ones((M2, N2)) * default_value

    # Apply mapping
    for k in tqdm(range(M)):
        for l in range(N):
            x = k + 0.5
            y = l + 0.5
            p = np.array([x, y, 1])
            q = np.matmul(A, p)
            [x, y, _] = q
            i = round(x - 0.5)
            j = round(y - 0.5)
            try:
                img2[i, j] = img[k, l]
            except:
                pass  # Out of index

    return img2
