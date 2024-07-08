"""https://github.com/mfaruqui/sparse-coding"""

from numpy import array, ndarray
from numpy import sum as np_sum
from numpy import abs as np_abs
from numpy.linalg import norm as np_norm

from utils_to_refactor import Param
from utils import read_dense_vectors_from_file, binarise_vectors_by_threshold, write_vectors_to_file


class Optimiser:
    def __init__(
        self, 
        vector_length:int, 
        vocabulary_size:int,
        n_iterations:int,
        sparse_vector_enlargement_factor:int=10,
        l1_regularisation_penalty:float=5e-1, 
        l2_regularisation_penalty:float=1e-5, 
        max_average_error:float=5e-2,
        max_average_error_delta:float=1e-3,
        learning_rate:float=5e-2,
        max_iterations:int=75,
    ) -> None:
        self.learning_rate=learning_rate
        self.n_iterations = n_iterations
        self.max_iterations = max_iterations
        self.max_average_error_delta = max_average_error_delta
        self.max_average_error = max_average_error
        self.l1_regularisation_penalty=l1_regularisation_penalty
        self.l2_regularisation_penalty=l2_regularisation_penalty 
        self.vec_len = vector_length
        self.factor = sparse_vector_enlargement_factor

        #TODO: update this when class modified
        self.dict = Param()
        self.dict.init(self.vec_len, self.factor * self.vec_len)
        self.sparse_positive_vectors = [Param() for _ in range(vocabulary_size)]
        for vec in self.sparse_positive_vectors:
            vec.init(self.factor * self.vec_len, 1)

    def forward_pass(self, vector_index:int) -> ndarray:
        return self.dict.var @ self.sparse_positive_vectors[vector_index].var

    def backward_pass(self, vector_index:int, difference_vector:ndarray) -> None:
        dict_grad = -2 * difference_vector @ self.sparse_positive_vectors[vector_index].var.T + 2 * self.l2_regularisation_penalty * self.dict.var
        self.dict.adagrad_update(self.learning_rate, dict_grad)
        atom_elem_grad = -2 * self.dict.var.T @ difference_vector
        self.sparse_positive_vectors[vector_index].adagrad_update_with_l1_reg_non_neg(self.learning_rate, atom_elem_grad, self.l1_regularisation_penalty)

    def sparsify_dense_vectors(self,dense_vectors:ndarray) -> ndarray:
        average_error,previous_average_error = 1,0
        for iteration in range(self.n_iterations):
            if self.stopping_condition(
                iteration=iteration,
                average_error=average_error,
                average_error_delta=abs(average_error - previous_average_error)
            ):
                break
            total_error,atom_l1_norm = 0,0
            for index, word_vector in enumerate(dense_vectors):
                predicted_vector = self.forward_pass(vector_index=index)
                diff_vector = word_vector - predicted_vector
                total_error += np_sum(diff_vector ** 2)
                atom_l1_norm += np_sum(np_abs(self.sparse_positive_vectors[index].var)) 
                self.backward_pass(
                    vector_index=index, 
                    difference_vector=diff_vector, 
                )
                print(f"\rProcessed words: {index}", end='')
            previous_average_error = average_error
            average_error = total_error / len(dense_vectors)
            print(f"\nIteration: {iteration}\nError per example: {average_error}\nDict L2 norm: {np_norm(self.dict.var)}\nAvg Atom L1 norm: {atom_l1_norm / len(dense_vectors)}")
        return self.sparse_vectors()
    
    def stopping_condition(
        self, 
        iteration:int,
        average_error:float, 
        average_error_delta:float
    ) -> bool:
        return iteration > self.max_iterations and average_error > self.max_average_error and average_error_delta > self.max_average_error_delta

    def sparse_vectors(self) -> ndarray:
        return array([vector.var.flatten() for vector in self.sparse_positive_vectors])
    



vocabulary = read_dense_vectors_from_file(filename="dense_vectors.txt")
dense_word_vectors = array(list(vocabulary.values()))
vector_size,vector_length,_ = dense_word_vectors.shape
optimiser = Optimiser(
    vector_length=vector_length, 
    vocabulary_size=vector_size,
    sparse_vector_enlargement_factor=3,
    n_iterations=20,
)
sparse_positive_word_vectors = optimiser.sparsify_dense_vectors(dense_vectors=dense_word_vectors)
binary_word_vectors = binarise_vectors_by_threshold(vectors=sparse_positive_word_vectors, threshold=0.0)
write_vectors_to_file(filename="binary_vectors.txt", vocabulary=list(vocabulary), vectors=binary_word_vectors)