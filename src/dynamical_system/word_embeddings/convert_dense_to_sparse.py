"""https://github.com/mfaruqui/sparse-coding"""

"""./sparse.o sample_vecs.txt 10 0.5 1e-5 1 out_vecs.txt"""
import numpy as np
import math
from collections import defaultdict

# Constants
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

def train(out_file, factor, cores, l1_reg, l2_reg, word_vecs, vocab):
    model = Model(factor, word_vecs[0].shape[0], len(word_vecs))
    avg_error = 1
    prev_avg_err = 0
    iter = 0
    while iter < 20 or (avg_error > 0.05 and iter < 75 and abs(avg_error - prev_avg_err) > 0.001):
        iter += 1
        print(f"\nIteration: {iter}")
        num_words = 0
        total_error = 0
        atom_l1_norm = 0
        for word_id, word_vec in enumerate(word_vecs):
            pred_vec = model.predict_vector(word_vec, word_id)
            diff_vec = word_vec - pred_vec
            error = np.sum(diff_vec ** 2)
            total_error += error
            num_words += 1
            atom_l1_norm += np.sum(np.abs(model.atom[word_id].var))
            model.update_params(word_id, RATE, diff_vec, l1_reg, l2_reg)
            print(f"\rProcessed words: {num_words}", end='')
        prev_avg_err = avg_error
        avg_error = total_error / num_words
        print(f"\nError per example: {avg_error}")
        print(f"Dict L2 norm: {np.linalg.norm(model.dict.var)}")
        print(f"Avg Atom L1 norm: {atom_l1_norm / num_words}")

    model.write_vectors_to_file(out_file, vocab)
    model.write_dict_to_file(out_file + "_dict")

def read_vecs_from_file(filename):
    vocab = defaultdict(str)
    word_vecs = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array([float(x) for x in parts[1:]]).reshape(-1, 1)
            vocab[len(vocab)] = word
            word_vecs.append(vec)
    return vocab, word_vecs

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 7:
        vec_corpus = sys.argv[1]
        factor = int(sys.argv[2])
        l1_reg = float(sys.argv[3])
        l2_reg = float(sys.argv[4])
        num_cores = int(sys.argv[5])
        outfilename = sys.argv[6]

        vocab, word_vecs = read_vecs_from_file(vec_corpus)

        print("Model specification")
        print("----------------")
        print(f"Vector length: {word_vecs[0].shape[0]}")
        print(f"Dictionary length: {factor * word_vecs[0].shape[0]}")
        print(f"L2 Reg (Dict): {l2_reg}")
        print(f"L1 Reg (Atom): {l1_reg}")
        print(f"Number of Cores: {num_cores}")
        print("----------------")

        train(outfilename, factor, num_cores, l1_reg, l2_reg, word_vecs, vocab)
    else:
        print(f"Usage: {sys.argv[0]} vec_corpus factor l1_reg l2_reg num_cores outfilename")
