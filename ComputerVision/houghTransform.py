import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import numpy as np
import math


class HoughTransform:
    """
    Apply Hough Transform
    """

    @staticmethod
    def hough_line(img, r=None):
        """
        Hough line detection
        :param img: image
        :param r: not use (default=None)
        :return: accumulator, p_max
        """
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
        """
        Hough circle detection
        :param img: image
        :param r: radio
        :return: accumulator N
        """
        # Inspired from Wikipedia
        (M, N) = img.shape
        index = np.array(list(zip(np.nonzero(img)))).T.reshape((-1, 2))
        theta = np.radians(np.arange(-90, 90))
        accumulator = np.zeros((M, N))
        for i in tqdm(index):
            x = i[0]
            y = i[1]
            for t in theta:
                b = int(y - r * np.sin(t))
                a = int(x - r * np.cos(t))
                if 0 <= a < M and 0 <= b < N:
                    accumulator[a, b] += 1
        return accumulator, N

    @staticmethod
    def draw_line(A, top_):
        """
        Draw top lines in the image
        :param A:
        :param top_:
        :return:
        """
        x = np.arange(0, A.shape[1])
        fig, ax = plt.subplots()
        ax.imshow(A, cmap='gray')
        for [theta, rho] in top_:
            y = (rho - x * math.cos(theta)) / math.sin(theta)
            ax.plot(y, x, '-', linewidth=1, color='red')
        plt.xlim(0, A.shape[1] - 1)

    @staticmethod
    def get_top_circles(acumulator, num_lines=10):
        """
        Get top circles of the image
        :param acumulator: acumulator
        :param num_lines: number of lines (default=10)
        :return: top circles
        """
        accumulator2 = np.copy(acumulator)
        top_ = []
        for _ in range(num_lines):
            a, b = np.unravel_index(np.argmax(accumulator2), accumulator2.shape)
            accumulator2[a, b] = 0
            top_.append([a, b])
        return top_

    @staticmethod
    def draw_circle(A, top_, r=7):
        """
        Draw circle
        :param A: image
        :param top_: top circles
        :param r: radius
        :return:
        """
        fig, ax = plt.subplots()
        circles = []
        for [x, y] in top_:
            circles.append(plt.Circle(xy=(x, y), ec='red', radius=r))
            circles[-1].set_facecolor('none')
            ax.add_patch(circles[-1])
        ax.imshow(A)

    @staticmethod
    def get_top_lines(accumulator, p_max, num_lines=10):
        """
        Get top lines
        :param accumulator: acumulator
        :param p_max: p max value
        :param num_lines: number of lines
        :return: top lines
        """
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
    def non_maxima_supresion(accumulator):
        """
        Non maxima supresion apply in accumulator
        :param accumulator: accumulator
        :return: new accumulator
        """
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
        """
        Apply Hough transform
        :param img: image
        :param r: radius (Only necessary for circle, default=7)
        :param method: method use ('line', 'circle', default='line')
        :param supresion: do supression in accumulator (default=True)
        :param plot: plot result (default=True)
        :param num_lines: number of lines or circles (default=True)
        :return: top results, acucmulator, None or supress accumulator
        """
        methods = {
            'line': HoughTransform.hough_line,
            'circle': HoughTransform.hough_circle
        }
        accumulator, p_max = methods[method](img, r)

        accumulator2 = None
        if supresion:
            accumulator2 = HoughTransform.non_maxima_supresion(accumulator)
        if method == 'line':
            top_ = HoughTransform.get_top_lines(accumulator2, p_max, num_lines=num_lines)
            if plot:
                HoughTransform.draw_line(img, top_)
        else:
            top_ = HoughTransform.get_top_circles(accumulator2, num_lines=num_lines)
            if plot:
                HoughTransform.draw_circle(img, top_, r=r)
        return top_, accumulator, accumulator2
