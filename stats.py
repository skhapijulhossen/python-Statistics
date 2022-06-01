import numpy as np
from collections import Counter

class Stats:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.row = data.shape[0]
        try:
            self.column = data.shape[1]
        except Exception:
            if self.row > 0:
                self.column = 0


    def mean(self):
        mean = [(np.sum(self.data[:, col]) / self.row)
                for col in range(self.column)]
        return np.array(mean)


    def median(self):
        medians = []
        try:
            for col in range(self.column):
                self.data[:, col] = np.sort(self.data[:, col])
                if self.row % 2 == 0:
                    index1 = self.row//2 - 1
                    index2 = index1 + 1
                    medians.append((self.data[index1, col]
                                    + self.data[index2, col])/2)
                else:
                    index = self.row//2 + 1
                    medians.append(self.data[index, col])
        except Exception:
            return -1
        return np.array(medians)


    def mode(self):
        modes = []
        try:
            for col in range(self.column):
                counter = {}
                for elem in self.data[:, col]:
                    if elem in counter.keys():
                        counter[elem] += 1
                    else:
                        counter[elem] = 1
                counter = Counter([(key, value) for key, value in counter.items()])
                frequency = sorted(counter, key=lambda tup:tup[1], reverse=True)[0][1]
                mode = [key[0] for key in counter.keys() if key[1]==frequency]
                modes.append(mode)
        except Exception as e:
            return (-1,e)
        return np.array(modes)


    def standardDev(self):
        stdDevs = []
        means = self.mean()
        for col in range(self.column):
            std = np.sum(pow(self.data[:, col] - means[col], 2))
            stdDevs.append(np.sqrt(std/(self.row-1)))
        return np.array(stdDevs)


    def skewness(self):
        skewNess = []
        means = self.mean()
        stds = self.standardDev()
        try:
            for col in range(self.column):
                skew = np.sum(pow(self.data[:, col] - means[col], 3))
                skew = (skew / pow(stds[col], 3)) / self.row
                skewNess.append(skew)
        except Exception as e:
            return (-1,e)
        return np.array(skewNess)


    def kurtosis(self):
        kurtosis = []
        means = self.mean()
        stds = self.standardDev()
        try:
            for col in range(self.column):
                kurtosisVal = np.sum(pow(self.data[:, col] - means[col], 4))
                kurtosisVal = (kurtosisVal / pow(stds[col], 4)) / self.row
                kurtosis.append(kurtosisVal)
        except Exception as e:
            return (-1,e)
        return np.array(kurtosis)


    def correlation(self):
        correlation = []
        try:
            for x in range(self.column):
                corr = []
                for y in range(self.column):
                    meanX = np.sum(self.data[:, x]) / self.row
                    meanY = np.sum(self.data[:, y]) / self.row
                    positive = 0
                    negetive = 0
                    if y != x:
                        for index in range(self.row):
                            if ((self.data[index, x] >= meanX) and (self.data[index, y] >= meanY)) \
                                    or ((self.data[index, x] < meanX) and (self.data[index, y] < meanY)):
                                positive += 1
                            else:
                                negetive += 1
                    if y != x:
                        ssxy = np.sum(
                            (self.data[:, x] - meanX) * (self.data[:, y] - meanY))
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
        except Exception as e:
            return (-1,e)
        return np.array(correlation)


if __name__ == '__main__':
    array = np.array([[2,6],
                    [2,7],
                    [5,7],
                    [5,7]])
    # array = np.array([[1, 2], [4, 9], [6, 7]])
    stats = Stats(array)
    print(stats.mean())
    print(stats.median())
    print(stats.standardDev())
    print(stats.skewness())
    print(stats.kurtosis())
    print(stats.correlation())
    print(stats.mode())
