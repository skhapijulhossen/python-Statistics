import numpy as np


class Stats:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.row = data.shape[0]
        self.column = data.shape[1]

    def mean(self):
        mean = [(np.sum(self.data[:, col]) / self.row)
                for col in range(self.column)]
        return np.array(mean)

    def median(self):
        medians = []
        for col in range(self.column):
            if self.row % 2 == 0:
                index1 = self.row//2 - 1
                index2 = index1 + 1
                medians.append(
                    (self.data[index1, col] + self.data[index2, col])/2)
            else:
                index = self.row//2 + 1
                medians.append(self.data[index, col])
        return np.array(medians)

    def standardDev(self):
        stdDevs = []
        means = self.mean()
        for col in range(self.column):
            std = np.sum(pow(self.data[:, col] - means[col], 2))
            stdDevs.append(np.sqrt(std/self.row))
        return np.array(stdDevs)

    def skewness(self):
        skewNess = []
        means = self.mean()
        stds = self.standardDev()
        for col in range(self.column):
            skew = np.sum(pow(self.data[:, col] - means[col], 3))
            skew = (skew / pow(stds[col], 3)) / self.row
            skewNess.append(skew)
        return np.array(skewNess)

    def kurtosis(self):
        kurtosis = []
        means = self.mean()
        stds = self.standardDev()
        for col in range(self.column):
            kurtosisVal = np.sum(pow(self.data[:, col] - means[col], 4))
            kurtosisVal = (kurtosisVal / pow(stds[col], 4)) / self.row
            kurtosis.append(kurtosisVal)
        return np.array(kurtosis)

    def correlation(self):
        correlation = []
        for x in range(self.column):
            corr = []
            for y in range(self.column):
                if y != x:
                    meanX = self.data[:, x] / self.row
                    meanY = self.data[:, y] / self.row
                    covariance = np.sum(
                        (self.data[:, x] - meanX) * (self.data[:, y] - meanY)) / np.sqrt(
                        np.sum(np.square(self.data[:, x] - meanX)) * np.sum(np.square(self.data[:, y] - meanY)))
                    corr.append(covariance)
                else:
                    corr.append(1)
            correlation.append(corr)
        return np.array(correlation)


if __name__ == '__main__':
    array = np.array([[1, 2, 44], [2, 5, 34], [3, 8, 24], [4, 5, 14]])
    stats = Stats(array)
    print(stats.mean())
    print(stats.median())
    print(stats.standardDev())
    print(stats.skewness())
    print(stats.kurtosis())
    print(stats.correlation())
