"""https://github.com/mfaruqui/sparse-coding"""
from numpy import array, ndarray
from pathlib import Path

#TODO: refactor
#TODO: double check refactored python file same as original C file



import numpy as np
import math



RHO = 0.95
EPSILON = 1e-6
RATE = 0.05

def sgn(val):
    return np.sign(val)

class Param:
    def __init__(self):
        self.var = None
        self._del_var = None
        self._del_grad = None
        self._grad_sum = None
        self._epsilon = None
        self._update_num = 0

    def init(self, rows, cols):
        if cols == 1:
            self.var = (0.6 / math.sqrt(rows)) * np.random.randn(rows, 1)
            self._del_var = np.zeros((rows, 1))
            self._del_grad = np.zeros((rows, 1))
            self._grad_sum = np.zeros((rows, 1))
            self._epsilon = EPSILON * np.ones((rows, 1))
        else:
            self.var = (0.6 / math.sqrt(rows + cols)) * np.random.randn(rows, cols)
            self._del_var = np.zeros((rows, cols))
            self._del_grad = np.zeros((rows, cols))
            self._grad_sum = np.zeros((rows, cols))
            self._epsilon = EPSILON * np.ones((rows, cols))

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
                    self.var[i, j] = -sgn(self._grad_sum[i, j]) * rate * diff / math.sqrt(self._del_grad[i, j] + EPSILON)

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
                    temp = -sgn(self._grad_sum[i, j]) * rate * diff / math.sqrt(self._del_grad[i, j] + EPSILON)
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

class Model:
    def __init__(self, times, vector_len, vocab_len):
        self.vec_len = vector_len
        self.factor = times
        self.dict = Param()
        self.dict.init(self.vec_len, self.factor * self.vec_len)
        self.atom = [Param() for _ in range(vocab_len)]
        for vec in self.atom:
            vec.init(self.factor * self.vec_len, 1)

    def non_linearity(self, vec):
        np.clip(vec, -1, 1, out=vec)

    def predict_vector(self, word_vec, word_index):
        return self.dict.var @ self.atom[word_index].var

    def update_params(self, word_index, rate, diff_vec, l1_reg, l2_reg):
        dict_grad = -2 * diff_vec @ self.atom[word_index].var.T + 2 * l2_reg * self.dict.var
        self.dict.adagrad_update(rate, dict_grad)
        atom_elem_grad = -2 * self.dict.var.T @ diff_vec
        self.atom[word_index].adagrad_update_with_l1_reg_non_neg(rate, atom_elem_grad, l1_reg)

    def write_vectors_to_file(self, filename, vocab):
        with open(filename, 'w') as outfile:
            for i, atom in enumerate(self.atom):
                word = vocab[i]
                vec_str = ' '.join(f"{val:.3f}" for val in atom.var.flatten())
                outfile.write(f"{word} {vec_str}\n")

    def write_dict_to_file(self, filename):
        self.dict.write_to_file(filename)


#TODO: continue refactoring from here
def optimise(
    word_vectors:list[ndarray], 
    vocabulary:list[str],
    factor:int, 
    l1_regularisation_penalty:float, 
    l2_regularisation_penalty:float, 
    output_file:str, 
) -> None:
    model = Model(factor, word_vectors[0].shape[0], len(word_vectors))
    avg_error = 1
    prev_avg_err = 0
    iter = 0
    while iter < 20 or (avg_error > 0.05 and iter < 75 and abs(avg_error - prev_avg_err) > 0.001):
        iter += 1
        print(f"\nIteration: {iter}")
        num_words = 0
        total_error = 0
        atom_l1_norm = 0
        for word_id, word_vec in enumerate(word_vectors):
            pred_vec = model.predict_vector(word_vec, word_id)
            diff_vec = word_vec - pred_vec
            error = np.sum(diff_vec ** 2)
            total_error += error
            num_words += 1
            atom_l1_norm += np.sum(np.abs(model.atom[word_id].var))
            model.update_params(word_id, RATE, diff_vec, l1_regularisation_penalty, l2_regularisation_penalty)
            print(f"\rProcessed words: {num_words}", end='')
        prev_avg_err = avg_error
        avg_error = total_error / num_words
        print(f"\nError per example: {avg_error}")
        print(f"Dict L2 norm: {np.linalg.norm(model.dict.var)}")
        print(f"Avg Atom L1 norm: {atom_l1_norm / num_words}")

    model.write_vectors_to_file(output_file, vocabulary)
    model.write_dict_to_file(output_file + "_dict")


        

input_file = Path("sample_vecs.txt")
vocabulary, word_vectors = [], []
with input_file.open() as f:
    for line in f.readlines():
        parts = line.strip().split()
        vocabulary.append(parts[0])
        word_vectors.append(array(list(map(float,parts[1:]))).reshape(-1,1))

optimise(
    output_file="out_vecs.txt", 
    factor=3, #10, #how much larger the embeddings will be
    l1_regularisation_penalty=0.5, 
    l2_regularisation_penalty=1e-5, 
    word_vectors=word_vectors, 
    vocabulary=vocabulary
)    
