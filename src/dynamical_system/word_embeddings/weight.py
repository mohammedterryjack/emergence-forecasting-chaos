#TODO: refactor

from numpy import zeros, ones, sign
from numpy import abs as np_abs
from numpy import sqrt as np_sqrt
from numpy.random import randn
from math import sqrt


class Weight:
    def __init__(self, rows:int, cols:int) -> None:
        self.var = None
        self._del_var = None
        self._del_grad = None
        self._grad_sum = None
        self._epsilon = None
        self._update_num = 0
        self.EPSILON = 1e-6

        if cols == 1:
            self.var = (0.6 / sqrt(rows)) * randn(rows, 1)
            self._del_var = zeros((rows, 1))
            self._del_grad = zeros((rows, 1))
            self._grad_sum = zeros((rows, 1))
            self._epsilon = self.EPSILON * ones((rows, 1))
        else:
            self.var = (0.6 / sqrt(rows + cols)) * randn(rows, cols)
            self._del_var = zeros((rows, cols))
            self._del_grad = zeros((rows, cols))
            self._grad_sum = zeros((rows, cols))
            self._epsilon = self.EPSILON * ones((rows, cols))

    def adagrad_update(self, rate, grad):
        self._del_grad += np_abs(grad) ** 2
        self._grad_sum += grad
        self.var -= rate * grad / np_sqrt(self._del_grad + self._epsilon)

    def adagrad_update_with_l1_reg_non_neg(self, rate, grad, l1_reg):
        self._update_num += 1
        self._del_grad += np_abs(grad) ** 2
        self._grad_sum += grad
        for i in range(self.var.shape[0]):
            for j in range(self.var.shape[1]):
                diff = abs(self._grad_sum[i, j]) - self._update_num * l1_reg
                if diff <= 0:
                    self.var[i, j] = 0
                else:
                    temp = -sign(self._grad_sum[i, j]) * rate * diff / sqrt(self._del_grad[i, j] + self.EPSILON)
                    if temp >= 0:
                        self.var[i, j] = temp
                    else:
                        self.var[i, j] = 0
