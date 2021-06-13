from ComputerVision.Values_Transformations.filter_kernel import gaussian_kernel
from ComputerVision.Values_Transformations.filter import Filter
from tqdm.notebook import tqdm
import numpy as np


class HarrisMatrix:

    def __init__(self, Gx, Gy, w):
        """
        Class of Harris Matrix
        :param Gx: Gradient x
        :param Gy: Gradient y
        :param w: weight kernel
        """
        self.Gxx = np.power(Gx, 2)
        self.Gyy = np.power(Gy, 2)
        self.Gxy = Gx * Gy
        self.w = w

    def full_R(self, k=0.05):
        """
        Return full matrix
        :param k: variable use
        :return: full matrix
        """
        R = np.zeros(self.Gxx.shape)
        for i in tqdm(range(R.shape[0])):
            for j in range(R.shape[1]):
                R[i, j] = np.linalg.det(self(i, j)) + \
                          k * np.power(np.sum(
                    self(i, j).diagonal()), 2)
        return R

    def __call__(self, s, t):
        """
        Return one pixel value
        :param s: coordinate x
        :param t: coordinate y
        :return: value
        """
        return self.w[s, t] * np.array([[self.Gxx[s, t], self.Gxy[s, t]],
                                        [self.Gxy[s, t], self.Gyy[s, t]]])


class HarrisStephens:
    # Regions by pixels

    @staticmethod
    def calculateGxGy(img):
        """
        Calculate Gradients Gx, Gy
        :param img: iamge
        :return: Gx, Gy
        """
        Gx = np.zeros(img.shape)
        Gy = np.zeros(img.shape)
        gx = np.array([-0.5, 0, 0.5])
        gy = np.array([-0.5, 0, 0.5])
        a = b = int((gx.shape[0] - 1) / 2)
        for i in tqdm(range(a, img.shape[0] - a)):
            for j in range(b, img.shape[1] - b):
                for s in range(-a, a + 1):
                    Gy[i, j] += gy[s] * img[i, j - s]
                for t in range(-b, b + 1):
                    Gx[i, j] += gx[t] * img[i - t, j]
        return Gx, Gy

    @staticmethod
    def matrix_m(s, t, Gx, Gy, w):
        """
        calculate matrix m value
        :param s: coordinate x
        :param t: coordinate y
        :param Gx: Gradient x
        :param Gy: Gradient y
        :param w: weight kernel
        :return: value
        """
        GxGy = Gx[s, t] * Gy[s, t]
        return w[s, t] * np.array([[Gx[s, t] ** 2, GxGy],
                                   [GxGy, Gy[s, t] ** 2]])

    @staticmethod
    def harrisStepherApprox(img, w, Gx, Gy, kernel_size=3):
        """
        Apply harries Stepher pixel-wise approximation
        :param img: image
        :param w: kernel
        :param Gx: Gradient x
        :param Gy: Gradient y
        :param kernel_size: size kernel
        :return: new image
        """
        img2 = np.zeros(img.shape)
        a = b = int((kernel_size - 1) / 2)
        for i in tqdm(range(a, img.shape[0] - a)):
            for j in range(b, img.shape[1] - b):
                img2[i, j] += w[i, j] * np.array([i, j]) @ \
                              HarrisStephens.matrix_m(i, j, Gx, Gy, w) @ \
                              np.array([[i], [j]])
        return img2

    @staticmethod
    def get_neighbours(i, j, shape_):
        """
        Get neighbours from pixel
        :param i: coordinate x
        :param j: coordinate y
        :param shape_: shape image
        :return: neighbours
        """
        list_indexes = [(i, j)]
        if i - 1 >= 0:
            if j - 1 >= 0:
                list_indexes.append((i - 1, j - 1))
            if j >= 0:
                list_indexes.append((i - 1, j))
            if j + 1 < shape_[1]:
                list_indexes.append((i - 1, j + 1))
        if j - 1 >= 0:
            list_indexes.append((i, j - 1))
        if j + 1 < shape_[1]:
            list_indexes.append((i, j + 1))
        if i + 1 < shape_[0]:
            if j - 1 >= 0:
                list_indexes.append((i + 1, j - 1))
            if j >= 0:
                list_indexes.append((i + 1, j))
            if j + 1 < shape_[1]:
                list_indexes.append((i + 1, j + 1))
        return list_indexes

    @staticmethod
    def non_maxima_supresion(img):
        """
        Non maxima supresion apply in img
        :param img: img
        :return: new img
        """
        img2 = np.zeros(img.shape)
        for i in tqdm(range(img.shape[0])):
            for j in range(img.shape[1]):
                key = True
                for (x, y) in HarrisStephens.get_neighbours(i, j,
                                                            img.shape):
                    if img[i][j] < img[x][y]:
                        key = False
                if key:
                    img2[i][j] = img[i][j]
        return img2

    @staticmethod
    def apply(img, t=0.01, k=0.05, kernel_size=3, gamma=3.5):
        """
        Apply corner detection
        :param img: image
        :param t: threshold 1
        :param k: threshold 2
        :param kernel_size: kernel size
        :param gamma: gamma value gaussian kernel
        :return: new image
        """
        print("Harris-Stephens corner detector")
        # Step 1
        print("    (1/5) Differentiation")
        Gx, Gy = HarrisStephens.calculateGxGy(img)
        # Step 2 - 3
        print("    (2-3/5) Pre-compute / Integration")
        gk = gaussian_kernel(kernel_size, gamma)
        w = Filter.apply_filter(img, gk)
        M = HarrisMatrix(Gx, Gy, w)
        # Step 4
        print("    (4/5) Compute R")
        R = M.full_R(k=k)
        # Step 5
        print("    (5/5) Apply threshold and non maxima supression")
        R_mn = HarrisStephens.non_maxima_supresion(R)
        R_n = R_mn / np.max(R_mn)
        f = np.vectorize(lambda x: 1 if x > t else 0)
        R2 = f(R_n)
        print("Complete!")
        return R2
