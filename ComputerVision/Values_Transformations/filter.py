from tqdm.notebook import tqdm
import numpy as np


class Filter:

    @staticmethod
    def add_bounds(img, kernel_size):
        num = int((kernel_size - 1) / 2)
        img2 = np.zeros(np.array(img.shape) + 2 * num)
        img2[1 * num: -1 * num, 1 * num:-1 * num] = img
        return img2.astype(int)

    @staticmethod
    def remove_bounds(img, kernel_size):
        num = int((kernel_size - 1) / 2)
        return img[1 * num: -1 * num, 1 * num:-1 * num]

    @staticmethod
    def correlation(img, g):
        a = int((g.shape[0] - 1) / 2)
        b = int((g.shape[1] - 1) / 2)
        img2 = np.zeros(img.shape)
        for i in tqdm(range(a, img.shape[0] - a)):
            for j in range(b, img.shape[1] - b):
                for s in range(-a, a + 1):
                    for t in range(-b, b + 1):
                        img2[i, j] += g[s, t] * img[i + s, j + t]
        return img2

    @staticmethod
    def convolution(img, g):
        a = int((g.shape[0] - 1) / 2)
        b = int((g.shape[1] - 1) / 2)
        img2 = np.zeros(img.shape)
        for i in tqdm(range(a, img.shape[0] - a)):
            for j in range(b, img.shape[1] - b):
                for s in range(-a, a + 1):
                    for t in range(-b, b + 1):
                        img2[i, j] += g[s, t] * img[i - s, j - t]
        return img2

    @staticmethod
    def apply_filter(img, g, method='convolution'):
        methods = {
            'correlation': Filter.correlation,
            'convolution': Filter.convolution
        }
        kernel_size = g.shape[0]
        img2 = Filter.add_bounds(img, kernel_size)
        img2 = methods[method](img2, g)
        return Filter.remove_bounds(img2, kernel_size)

    @staticmethod
    def apply_operators(img, gx, gy, method='convolution'):
        methods = {
            'correlation': Filter.correlation,
            'convolution': Filter.convolution
        }
        kernel_size = g.shape[0]
        img2 = Filter.add_bounds(img, kernel_size)
        img2x = methods[method](img2, gx)
        img2y = methods[method](img2, gy)
        img2 = np.sqrt(img2x ** 2 + img2y ** 2)
        return Filter.remove_bounds(img2, kernel_size)

    @staticmethod
    def correlation_l(img, r, c):
        a = int((c.shape[0] - 1) / 2)
        img2 = np.zeros(img.shape)
        for i in tqdm(range(a, img.shape[0] - a)):
            for j in range(a, img.shape[1] - a):
                for s in range(-a, a + 1):
                    img2[i, j] += c[s] * img[i + s, j]
        img3 = np.zeros(img.shape)
        for i in tqdm(range(a, img.shape[0] - a)):
            for j in range(a, img.shape[1] - a):
                for t in range(-a, a + 1):
                    img3[i, j] += r[t] * img2[i, j + t]
        return img3

    @staticmethod
    def convolution_l(img, r, c):
        a = int((c.shape[0] - 1) / 2)
        img2 = np.zeros(img.shape)
        for i in tqdm(range(a, img.shape[0] - a)):
            for j in range(a, img.shape[1] - a):
                for s in range(-a, a + 1):
                    img2[i, j] += c[s] * img[i - s, j]
        img3 = np.zeros(img.shape)
        for i in tqdm(range(a, img.shape[0] - a)):
            for j in range(a, img.shape[1] - a):
                for t in range(-a, a + 1):
                    img3[i, j] += r[t] * img2[i, j - t]
        return img3

    @staticmethod
    def apply_filter_l(img, r, c, method='convolution'):
        methods = {
            'correlation': Filter.correlation_l,
            'convolution': Filter.convolution_l
        }
        kernel_size = r.shape[0]
        img2 = Filter.add_bounds(img, kernel_size)
        img2 = methods[method](img2, r, c)
        return Filter.remove_bounds(img2, kernel_size)

    @staticmethod
    def apply_operators_l(img, gx_r, gx_c, gy_r,
                          gy_c, method='convolution'):
        methods = {
            'correlation': Filter.correlation_l,
            'convolution': Filter.convolution_l
        }
        kernel_size = gx_r.shape[0]
        img2 = Filter.add_bounds(img, kernel_size)
        img2x = methods[method](img2, gx_r, gx_c)
        img2y = methods[method](img2, gy_r, gy_c)
        img2 = np.sqrt(img2x ** 2 + img2y ** 2)
        return Filter.remove_bounds(img2, kernel_size)
