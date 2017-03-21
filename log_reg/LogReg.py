import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import logistic


class LogReg:
    def __init__(self, X, Z, learning_rate=0.001, epochs=10000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.X = X
        self.Z = Z
        self.W = np.random.randn(1, 2) * 0.01
        self.b = np.random.randn(1) * 0.01

    def pred(self, W, X, b):
        return logistic.cdf(np.dot(X, W.T) + b)

    def __loglikelihood(self, X, Z, W, b):
        Y = self.pred(W, X, b)
        return -np.sum(Z.reshape(Z.size,1) * np.log(Y) + (1 - Z.reshape(Z.size,1)) * np.log(1 - Y), 0)

    def __grad(self, X, Z, W, b):
        dW = np.sum((logistic.cdf(np.dot(X, W.T) - Z.reshape(Z.size,1)) * X))
        db = np.sum((logistic.cdf(b - Z.reshape(Z.size,1)) * X))
        return dW, db

    def train(self):


        train_loss = []
        for i in range(self.epochs):
            dLdW, dLdb = self.__grad(self.X, self.Z, self.W, self.b)

            self.W -= self.learning_rate * dLdW
            self.b -= self.learning_rate * dLdb
            train_loss.append(- self.__loglikelihood(self.X, self.Z, self.W, self.b).mean())
        return train_loss

    def plot_decision_boundary(self, X, Z):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        # ax.scatter(X[:, 0], X[:, 1], c=Z, cmap=plt.cm.cool)
        ax.set_autoscale_on(False)

        a = - self.W[0, 0] / self.W[0, 1]
        xx = np.linspace(-30, 30)
        yy = a * xx - (self.b[0]) / self.W[0, 1]

        ax.plot(xx, yy )
        plt.interactive(False)
        plt.show()
