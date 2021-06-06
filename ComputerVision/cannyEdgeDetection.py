from ComputerVision.Values_Transformations.filter_kernel import sobel_operators, box_kernel
from ComputerVision.Values_Transformations.filter import Filter
from tqdm.notebook import tqdm
import numpy as np


class CannyEdge:

    @staticmethod
    def group_angle(x):
        if -22.5 <= x <= 22.5 or 157.5 <= x or -157.5 >= x:
            return 0
        if 22.5 <= x <= 67.5 or -112.5 >= x >= -157.5:
            return 45
        if 67.5 <= x <= 112.5 or -67.5 >= x >= -112.5:
            return 90
        if 112.5 <= x <= 157.5 or -22.5 >= x >= -67.5:
            return 135

    @staticmethod
    def compare_magnitude(x, y, i, j):
        if x == 0:
            if y[i, j] < y[i - 1, j] or y[i, j] < y[i + 1, j]:
                return 0
        if x == 45:
            if y[i, j] < y[i - 1, j + 1] or y[i, j] < y[i + 1, j - 1]:
                return 0
        if x == 90:
            if y[i, j] < y[i - 1, j] or y[i, j] < y[i + 1, j]:
                return 0
        if x == 135:
            if y[i, j] < y[i - 1, j] or y[i, j] < y[i + 1, j]:
                return 0
        return y[i, j]

    @staticmethod
    def compare_magnitudes(x, y):
        aux = np.zeros(x.shape)
        for i in tqdm(range(x.shape[0])):
            for j in range(x.shape[1]):
                try:
                    aux[i, j] = CannyEdge.compare_magnitude(x[i, j], y, i, j)
                except IndexError:
                    pass
        return aux

    @staticmethod
    def thresholding(x, th, tl):
        if x >= th:
            return 2
        if th > x >= tl:
            return 1
        return 0

    @staticmethod
    def get_neighbours(i, j, shape_):
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
    def region_grow(f, f5):
        while True:
            xy = np.nonzero(f(f5))
            if len(xy[0]) > 0:
                x = xy[0][0]
                y = xy[1][0]
                num = 0
                for (i, j) in CannyEdge.get_neighbours(x, y, f5.shape):
                    if f5[i, j] == 1:
                        num += 1
                        f5[i, j] = 2
                if num == 0:
                    f[x, y] = 0
            else:
                break
        return f5

    @staticmethod
    def apply(img, tl=0.05, th=0.15, do_grow=True, kernel_size=3):
        print("Applying Canny edge detector")
        # Step 1
        print("    (1/5) Integration")
        g = box_kernel(kernel_size)
        f1 = Filter.apply_filter(img, g)

        # Step 2
        print("    (2/5) Differentiation")
        # Sobbel
        g_x, g_y = sobel_operators()
        f2_x = Filter.apply_filter(f1, g_x)
        f2_y = Filter.apply_filter(f1, g_y)

        # Magnitude
        G = np.sqrt(f2_x ** 2 + f2_y ** 2)
        # Orientation
        # to don't divide by 0
        f = np.vectorize(lambda x: 10e-19 if x == 0. else x)
        teta = np.degrees(np.arctan(f(f2_y) / f(f2_x)))

        # Step 3
        print("    (3/5) Non-maxima suppression")
        # Group angles
        f = np.vectorize(lambda x: CannyEdge.group_angle(x))
        teta2 = f(teta)
        f3 = CannyEdge.compare_magnitudes(teta2, G)

        # Step 4
        print("    (4/5) Hysteresis (or Dual) thresholding")
        f = np.vectorize(CannyEdge.thresholding)
        f3n = f3 / np.max(f3)  # Normalize
        f4 = f(f3n, th, tl)

        # Step 5
        print("    (5/5) Connectivity analysis")
        f = np.vectorize(lambda x: 0 if x == 2 else x)
        f5 = np.copy(f4)
        if do_grow:
            f5 = CannyEdge.region_grow(f, f5)
        print("Complete!")
        return f5
