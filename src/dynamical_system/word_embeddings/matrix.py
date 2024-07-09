from numpy import zeros, sign, ndarray
from numpy import abs as np_abs
from numpy import sqrt as np_sqrt
from numpy.random import randn
from math import sqrt

class SparseMatrix:
    def __init__(self, width:int, height:int, epsilon:float=1e-6, learning_rate:float=5e-2) -> None:
        self.shape=(width,height)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.values = self.random_values(width=width,height=height)
        self.gradient_delta = zeros(shape=self.shape)

    def update(self, gradient:ndarray) -> None:
        self.gradient_delta += np_abs(gradient) ** 2
        self.values -= self.learning_rate * gradient / np_sqrt(self.gradient_delta + self.epsilon)

    @staticmethod
    def random_values(width:int,height:int,offset:float=0.6) -> float:
        return (offset / sqrt(width + height)) * randn(width, height)

class NonNegativeSparseMatrix(SparseMatrix):
    def __init__(self, width:int, height:int, l1_regularisation_penalty:float=5e-1) -> None:
        super().__init__(width=width, height=height)
        self.penalty = l1_regularisation_penalty
        self.gradient_sum = zeros(shape=self.shape)
        self.update_number = 0
        
    def update(self, gradient:ndarray) -> None:
        self.update_number += 1
        self.gradient_delta += np_abs(gradient) ** 2
        self.gradient_sum += gradient
        width,height = self.shape
        for i in range(width):
            for j in range(height):
                delta = abs(self.gradient_sum[i, j]) - self.update_number * self.penalty
                temp=-sign(self.gradient_sum[i, j]) * self.learning_rate * delta / sqrt(self.gradient_delta[i, j] + self.epsilon)
                self.values[i,j] = 0 if delta <= 0 else temp if temp >= 0 else 0