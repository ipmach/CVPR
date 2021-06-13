from tqdm.notebook import tqdm
import numpy as np


class Filter:
    """
    Filter Transformations
    """

    @staticmethod
    def add_bounds(img, kernel_size):
        """
        Add bounds to image (zero bound)
        :param img: image
        :param kernel_size: size kernel use
        :return: new img
        """
        num = int((kernel_size - 1) / 2)
        img2 = np.zeros(np.array(img.shape) + 2 * num)
        img2[1 * num: -1 * num, 1 * num:-1 * num] = img
        return img2.astype(int)

    @staticmethod
    def remove_bounds(img, kernel_size):
        """
        Remove bounds image
        :param img: image
        :param kernel_size: size kernel used
        :return: new img
        """
        num = int((kernel_size - 1) / 2)
        return img[1 * num: -1 * num, 1 * num:-1 * num]

    @staticmethod
    def correlation(img, g):
        """
        Apply correlation
        :param img: image
        :param g: kernel
        :return: new image
        """
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
        """
        Apply convolution
        :param img: image
        :param g: kernel
        :return: new image
        """
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
        """
        Apply filter transformation
        :param img: image
        :param g: kernel
        :param method: method use ('correlation', 'convolution', default='convolution')
        :return: new image
        """
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
        """
        Apply operators transformation
        :param img: image
        :param gx: kernel x axis
        :param gy: kernel y axis
        :param method: method use ('correlation', 'convolution', default='convolution')
        :return: new image
        """
        methods = {
            'correlation': Filter.correlation,
            'convolution': Filter.convolution
        }
        kernel_size = gx.shape[0]
        img2 = Filter.add_bounds(img, kernel_size)
        img2x = methods[method](img2, gx)
        img2y = methods[method](img2, gy)
        img2 = np.sqrt(img2x ** 2 + img2y ** 2)
        return Filter.remove_bounds(img2, kernel_size)

    @staticmethod
    def correlation_l(img, r, c):
        """
        Apply correlation linear separation
        :param img: image
        :param r: kernel 1
        :param c: kernel 2
        :return: new image
        """
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
        """
        Apply convolution linear separation
        :param img: image
        :param r: kernel 1
        :param c: kernel 2
        :return: new image
        """
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
        """
        Apply filter transformation linear separation
        :param img: image
        :param r: kernel 1
        :param c: kernel 2
        :param method: method use ('correlation', 'convolution', default='convolution')
        :return: new image
        """
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
        """
        Apply operators transformation linear separation
        :param img: image
        :param gx_r: kernel 1 x axis
        :param gx_c: kernel 2 x axis
        :param gy_r: kernel 1 y axis
        :param gy_c: kernel 2 y axis
        :param method: method use ('correlation', 'convolution', default='convolution')
        :return: new image
        """
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
