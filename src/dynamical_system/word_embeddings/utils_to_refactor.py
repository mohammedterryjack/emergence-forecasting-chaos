#TODO: refactor

import numpy as np
import math


class Param:
    def __init__(self):
        self.var = None
        self._del_var = None
        self._del_grad = None
        self._grad_sum = None
        self._epsilon = None
        self._update_num = 0
        self.EPSILON = 1e-6

    def init(self, rows, cols):
        if cols == 1:
            self.var = (0.6 / math.sqrt(rows)) * np.random.randn(rows, 1)
            self._del_var = np.zeros((rows, 1))
            self._del_grad = np.zeros((rows, 1))
            self._grad_sum = np.zeros((rows, 1))
            self._epsilon = self.EPSILON * np.ones((rows, 1))
        else:
            self.var = (0.6 / math.sqrt(rows + cols)) * np.random.randn(rows, cols)
            self._del_var = np.zeros((rows, cols))
            self._del_grad = np.zeros((rows, cols))
            self._grad_sum = np.zeros((rows, cols))
            self._epsilon = self.EPSILON * np.ones((rows, cols))

    def adagrad_update(self, rate, grad):
        self._del_grad += np.abs(grad) ** 2
        self._grad_sum += grad
        self.var -= rate * grad / np.sqrt(self._del_grad + self._epsilon)

    def adagrad_update_with_l1_reg(self, rate, grad, l1_reg):
        self._update_num += 1
        self._del_grad += np.abs(grad) ** 2
        self._grad_sum += grad
        for i in range(self.var.shape[0]):
            for j in range(self.var.shape[1]):
                diff = abs(self._grad_sum[i, j]) - self._update_num * l1_reg
                if diff <= 0:
                    self.var[i, j] = 0
                else:
                    self.var[i, j] = -np.sign(self._grad_sum[i, j]) * rate * diff / math.sqrt(self._del_grad[i, j] + self.EPSILON)

    def adagrad_update_with_l1_reg_non_neg(self, rate, grad, l1_reg):
        self._update_num += 1
        self._del_grad += np.abs(grad) ** 2
        self._grad_sum += grad
        for i in range(self.var.shape[0]):
            for j in range(self.var.shape[1]):
                diff = abs(self._grad_sum[i, j]) - self._update_num * l1_reg
                if diff <= 0:
                    self.var[i, j] = 0
                else:
                    temp = -np.sign(self._grad_sum[i, j]) * rate * diff / math.sqrt(self._del_grad[i, j] + self.EPSILON)
                    if temp >= 0:
                        self.var[i, j] = temp
                    else:
                        self.var[i, j] = 0

    def write_to_file(self, filename):
        with open(filename, 'w') as f:
            f.write(f"{self.var.shape[0]} {self.var.shape[1]} ")
            for i in range(self.var.shape[0]):
                for j in range(self.var.shape[1]):
                    f.write(f"{self.var[i, j]} ")
            f.write("\n")

    def read_from_file(self, filename):
        with open(filename, 'r') as f:
            line = f.readline().strip()
            data = line.split()
            rows = int(data[0])
            cols = int(data[1])
            self.var = np.zeros((rows, cols))
            idx = 2
            for i in range(rows):
                for j in range(cols):
                    self.var[i, j] = float(data[idx])
                    idx += 1
