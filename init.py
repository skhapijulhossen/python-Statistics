import numpy as np


class Stats:
    """
    This Class is implementation of Statistics concepts from scratch. Numpy is used for efficient N-Dimentional array Execution.
    """

    def __init__(self, data: np.ndarray):
        self.data = data
        self.row = data.shape[0]
        self.column = data.shape[1]

    def mean(self):
        "As Name suggest this function will return mean of N dimensional array."
        mean = [(np.sum(self.data[:, col]) / self.row)
                for col in range(self.column)]
        return np.array(mean)

    def median(self):
        "This Function will return Median of N-dimensional array."
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
        "This Function will return Standard deviation of N-dimensional array."
        stdDevs = []
        means = self.mean()
        for col in range(self.column):
            std = np.sum(pow(self.data[:, col] - means[col], 2))
            stdDevs.append(np.sqrt(std/self.row))
        return np.array(stdDevs)

    def skewness(self):
        "This Function will return Skewness of N-dimensional array."
        skewNess = []
        means = self.mean()
        stds = self.standardDev()
        for col in range(self.column):
            skew = np.sum(pow(self.data[:, col] - means[col], 3))
            skew = (skew / pow(stds[col], 3)) / self.row
            skewNess.append(skew)
        return np.array(skewNess)

    def kurtosis(self):
        "This Function will return Kurtosis value of N-dimensional array."
        kurtosis = []
        means = self.mean()
        stds = self.standardDev()
        for col in range(self.column):
            kurtosisVal = np.sum(pow(self.data[:, col] - means[col], 4))
            kurtosisVal = (kurtosisVal / pow(stds[col], 4)) / self.row
            kurtosis.append(kurtosisVal)
        return np.array(kurtosis)

    def correlation(self):
        "This Function will return Correlation of N-dimensional array in the form of matrix for each column."
        correlation = []
        for x in range(self.column):
            corr = []
            for y in range(self.column):
                meanX = np.sum(self.data[:, x]) / self.row
                meanY = np.sum(self.data[:, y]) / self.row
                positive = 0
                negetive = 0
                if y != x:
                    for index in range(self.row):
                        if ((self.data[index, x] >= meanX) and (self.data[index, y] >= meanY)) or ((self.data[index, x] < meanX) and (self.data[index, y] < meanY)):
                            positive += 1
                        else:
                            negetive += 1
                if y != x:
                    ssxy = np.sum((self.data[:, x] - meanX)
                                  * (self.data[:, y] - meanY))
                    ssxx = np.sum((self.data[:, x] - meanX)**2)
                    ssyy = np.sum((self.data[:, y] - meanY)**2)
                    covariance = ssxy / np.sqrt(ssxx * ssyy)
                    if positive >= negetive:
                        corr.append(+covariance)
                    else:
                        corr.append(-covariance)

                else:
                    corr.append(1)
            correlation.append(corr)
        return np.array(correlation)


if __name__ == '__main__':
    array = np.array([[x, np.sqrt((x**0.54)/(x*(x+2))), x**3]
                      for x in range(1, 20)])
    stats = Stats(array)
    print(stats.mean())
    print(stats.median())
    print(stats.standardDev())
    print(stats.skewness())
    print(stats.kurtosis())
    print(stats.correlation())
