class Interpolations:
    """
    Interpolations apply in inverse mappings
    """

    @staticmethod
    def no_inter(img, M, N, x, y):
        """
        No interpolation apply in pixel
        :param img: original image
        :param M: size image x
        :param N: size image y
        :param x: coordinate x
        :param y: coordinate y
        :return:  return interpolation
        """
        k = round(x) - 1
        l = round(y) - 1
        if (k >= 0) and (k < M - 1) and (l >= 0) and (l < N - 1):
            return img[k, l]

    @staticmethod
    def bilinear(img, M, N, x, y):
        """
        Bilinaer interpolation apply in pixel
        :param img: original image
        :param M: size image x
        :param N: size image y
        :param x: coordinate x
        :param y: coordinate y
        :return:  return interpolation
        """
        k = round(x) - 1
        l = round(y) - 1
        u = x - k - 0.5
        v = y - l - 0.5
        if (k >= 0) and (k < M - 1) and (l >= 0) and (l < N - 1):
            return round((1 - v) * ((1 - u) * img[k, l] + u * img[k + 1, l])
                         + v * ((1 - u) * img[k, l + 1] + u * img[k + 1, l + 1]))

    @staticmethod
    def nearest_neighbour(img, M, N, x, y):
        """
        Nearest neighbour interpolation apply in pixel
        :param img: original image
        :param M: size image x
        :param N: size image y
        :param x: coordinate x
        :param y: coordinate y
        :return:  return interpolation
        """
        k = round(x) - 1
        l = round(y) - 1
        if (k >= 0) and (k < M - 1) and (l >= 0) and (l < N - 1):
            return img[k, l]
