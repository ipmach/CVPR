import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import numpy as np


class Histogram:
    """
    Apply Histogram Transformations
    """

    @staticmethod
    def calculate_cdf(count):
        """
        Calculate cumulative distribution function
        :param count: count histogram
        :return: cumulative distribution function
        """
        cdf_ = [0]
        for i in range(1, len(count)):
            cdf_.append(cdf_[i - 1] + count[i])
        return cdf_

    @staticmethod
    def plot_histogram(hist, cdf=True, figsize=(15, 5)):
        """
        Plot histogram
        :param hist: Histogram
        :param cdf: Calculate cdf (default=True)
        :param figsize: size figure
        :return:
        """
        (values, count) = hist
        plt.figure(figsize=figsize)
        if cdf:
            plt.subplot(1, 2, 1)
            plt.bar(values, count)
            plt.subplot(1, 2, 2)
            cdf_ = Histogram.calculate_cdf(count)
            plt.plot(values, cdf_, 'r')
        else:
            plt.bar(values, count)
        plt.show()

    @staticmethod
    def histogram(img, visualize=True, normalize=False,
                  full=False, L=255, figsize=(15, 5)):
        """
        Creaet histogram from image intensities
        :param img: image
        :param visualize: do visualization (default=True)
        :param normalize: do normalization (default=False)
        :param full: do all columns even empty ones (default=False)
        :param L: Maximum intensity
        :param figsize: size figure
        :return: values, count
        """
        values, count = np.unique(img, return_counts=True)
        if full:  # Complete discrete space
            values2 = np.zeros(L)
            count2 = np.zeros(L)
            values2[values] = values
            count2[values] = count
            values = values2
            count = count2
        if normalize:
            # works for Gray and color
            total = 1
            for i in img.shape:
                total *= i
            count = count / total
        if visualize:
            Histogram.plot_histogram((values, count), figsize=figsize)
        return values, count

    @staticmethod
    def equalization_book(img, L=255, round_=True):
        """
        Book's equalization method
        :param img: image
        :param L: Maximum intensity
        :param round_: round values (default=True)
        :return: equalize image
        """
        (_, prob) = Histogram.histogram(img, visualize=False,
                                        normalize=True, full=True)
        f = np.vectorize(lambda k: (L - 1) * np.sum([prob[j]
                                                     for j in range(k)]))
        if round_:
            return np.round(f(img)).astype(int)
        else:
            return f(img)

    @staticmethod
    def equalization_wiki(img, L=255):
        """
        Wikipedia's equalization method
        :param img: image
        :param L: Maximum intensity
        :return: equalize image
        """
        img2 = Histogram.equalization_book(img, L=L)
        s_min = np.min(img2)
        f = np.vectorize(lambda s: (s - s_min) /
                                   (1 - (s_min / (L - 1))))
        return np.round(f(img2)).astype(int)

    @staticmethod
    def equalization_prof(img, L=255):
        """
        Profesor's equalization method
        :param img: image
        :param L: Maximum intensity
        :return: equalize image
        """
        (_, prob) = Histogram.histogram(img, visualize=False,
                                        normalize=True, full=True)
        f = np.vectorize(lambda k: (L - 1) * np.sum([prob[j]
                                                     for j in range(k)])
                                   + 0.5 * prob[k] - 0.5)
        return np.round(f(img)).astype(int)

    @staticmethod
    def equalization(img, L=255, method='prof'):
        """
        Apply equalization transformation
        :param img: image
        :param L: Maximum intensity
        :param method: method used ('book', 'wiki', 'prof', default='prof')
        :return: equalize image
        """
        methods = {
            'book': Histogram.equalization_book,
            'wiki': Histogram.equalization_wiki,
            'prof': Histogram.equalization_prof
        }
        return methods[method](img, L=L)

    @staticmethod
    def matching(img1, img2):
        """
        Apply matching transformation
        :param img1: original image
        :param img2: second image
        :return: new image
        """
        # Getting info need it
        (_, count1) = Histogram.histogram(img1, normalize=True,
                                          full=True, visualize=False)
        cdf1 = Histogram.calculate_cdf(count1)
        (_, count2) = Histogram.histogram(img2, normalize=True,
                                          full=True, visualize=False)
        cdf2 = Histogram.calculate_cdf(count2)

        # Appling matching
        img3 = np.zeros(img1.shape)
        for i in tqdm(range(img1.shape[0])):
            for j in range(img1.shape[1]):
                k = img1[i, j]
                z = 0.5 * (cdf1[k - 1] + cdf1[k])
                m = 0
                while cdf2[m + 1] < z:
                    m += 1
                Ls = m + (z - cdf2[m]) / count2[m]
                img3[i, j] = np.round(Ls - 0.5)

        return img3