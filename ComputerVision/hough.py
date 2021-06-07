import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import numpy as np
import math


class HoughTransform:

    @staticmethod
    def hough_line(img, r):
        (M, N) = img.shape
        index = np.array(list(zip(np.nonzero(img)))).T.reshape((-1, 2))

        p_max = int(np.round(np.sqrt(M ** 2 + N ** 2)))
        theta = np.radians(np.arange(-90, 90))
        accumulator = np.zeros((len(theta), 2 * p_max))

        for i in tqdm(index):
            x = i[0]
            y = i[1]
            p_values = np.round(x * np.cos(theta) + y * np.sin(theta))
            coord_acc = np.array(list(zip(np.arange(len(theta)), p_max + p_values))).astype(int)
            for [x, y] in coord_acc:
                accumulator[x][y] += 1
        return accumulator, p_max

    @staticmethod
    def hough_circle(img, r):
        raise Exception("Not working")
        (M, N) = img.shape
        index = np.array(list(zip(np.nonzero(img)))).T.reshape((-1, 2))

        a = np.arange(M)
        accumulator = np.zeros(img.shape)

        for i in tqdm(index):
            x = i[0]
            y = i[1]
            # p_values = np.round(x * np.cos(theta) + y * np.sin(theta))
            b = - 2 * y + np.sqrt(4 * (y ** 2) - 4 * ((x - a) ** 2 + y ** 2 - r ** 2))
            coord_acc = np.array(list(zip(np.arange(len(a)), b))).astype(int)
            for [x, y] in coord_acc:
                print(x, y)
                if 0 <= y < N:
                    accumulator[x][y] += 1
            b = - 2 * y - np.sqrt(4 * (y ** 2) - 4 * ((x - a) ** 2 + y ** 2 - r ** 2))
            coord_acc = np.array(list(zip(np.arange(len(a)), b))).astype(int)
            for [x, y] in coord_acc:
                if 0 <= y < N:
                    accumulator[x][y] += 1
        return accumulator, N

    @staticmethod
    def draw_line(A, top_):
        x = np.arange(0, A.shape[1])
        fig, ax = plt.subplots()
        ax.imshow(A, cmap='gray')
        for [theta, rho] in top_:
            y = (rho - x * math.cos(theta)) / math.sin(theta)
            ax.plot(y, x, '-', linewidth=1, color='red')
        plt.xlim(0, A.shape[1] - 1)

    @staticmethod
    def get_top_lines(accumulator, p_max, num_lines=10):
        accumulator2 = np.copy(accumulator)
        top_ = []
        for i in range(num_lines):
            (theta, p) = np.unravel_index(np.argmax(accumulator2), accumulator2.shape)
            accumulator2[theta][p] = 0
            theta, p = np.radians(theta - 90), p - p_max
            top_.append([theta, p])
        return top_

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
    def non_maxima_supresion(accumulator):
        accumulator2 = np.zeros(accumulator.shape)
        for i in tqdm(range(accumulator.shape[0])):
            for j in range(accumulator.shape[1]):
                key = True
                for (x, y) in HoughTransform.get_neighbours(i, j,
                                                            accumulator.shape):
                    if accumulator[i][j] < accumulator[x][y]:
                        key = False
                if key:
                    accumulator2[i][j] = accumulator[i][j]
        return accumulator2

    @staticmethod
    def apply(img, r=7, method='line', supresion=True, plot=True, num_lines=15):
        methods = {
            'line': HoughTransform.hough_line,
            'circle': HoughTransform.hough_circle
        }
        accumulator, p_max = methods[method](img, r)

        accumulator2 = None
        if supresion:
            accumulator2 = HoughTransform.non_maxima_supresion(accumulator)
        top_ = HoughTransform.get_top_lines(accumulator2, p_max, num_lines=num_lines)

        if plot:
            HoughTransform.draw_line(img, top_)
        return top_, accumulator, accumulator2