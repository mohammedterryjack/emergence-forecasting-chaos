from numpy import zeros, sign, ndarray
from numpy.random import randn

class SparseMatrix:
    def __init__(self, width:int, height:int, learning_rate:float=5e-2) -> None:
        self.shape=(width,height)
        self.learning_rate = learning_rate
        self.values = self.random_weights(width=width, height=height)
        self.gradient_delta = zeros(shape=self.shape)

    def update(self, gradient:ndarray) -> None:
        self.gradient_delta += gradient ** 2
        self.values -= self.learning_rate * gradient / (self.gradient_delta**0.5 + 1e-6)

    @staticmethod
    def random_weights(width:int, height:int) -> ndarray:
        return (0.6 / (width + height)**0.5) * randn(width, height) 

class NonNegativeSparseMatrix(SparseMatrix):
    def __init__(self, width:int, height:int, l1_regularisation_penalty:float=5e-1) -> None:
        super().__init__(width=width, height=height)
        self.penalty = l1_regularisation_penalty
        self.gradient_sum = zeros(shape=self.shape)
        self.update_number = 0
        
    def update(self, gradient:ndarray) -> None:
        self.update_number += 1
        self.gradient_delta += gradient ** 2
        self.gradient_sum += gradient
        width,height = self.shape
        for i in range(width):
            for j in range(height):
                delta = abs(self.gradient_sum[i, j]) - self.update_number * self.penalty
                temp = -sign(self.gradient_sum[i, j]) * self.learning_rate * delta / (self.gradient_delta[i, j]**0.5 + 1e-6)
                self.values[i,j] = 0 if delta <= 0 else temp if temp >= 0 else 0