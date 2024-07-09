from numpy import array, ndarray
from numpy import sum as np_sum
from numpy import abs as np_abs
from numpy.linalg import norm as np_norm

from matrix import SparseMatrix, NonNegativeSparseMatrix

class Optimiser:
    def __init__(
        self, 
        vector_length:int, 
        vocabulary_size:int,
        n_iterations:int,
        sparse_vector_enlargement_factor:int=10, 
        l2_regularisation_penalty:float=1e-5, 
        max_average_error:float=5e-2,
        max_average_error_delta:float=1e-3,
        max_iterations:int=75,
    ) -> None:
        self.dictionary_vectors = SparseMatrix(
            width=vector_length,
            height=sparse_vector_enlargement_factor * vector_length,
        )
        self.sparse_positive_vectors = [
            NonNegativeSparseMatrix(
                width=sparse_vector_enlargement_factor * vector_length,
                height=1,
            ) for _ in range(vocabulary_size)
        ]

        self.n_iterations = n_iterations
        self.max_iterations = max_iterations
        self.max_average_error_delta = max_average_error_delta
        self.max_average_error = max_average_error
        self.l2_regularisation_penalty=l2_regularisation_penalty 

    def forward_pass(self, vector_index:int) -> ndarray:
        return self.dictionary_vectors.values @ self.sparse_positive_vectors[vector_index].values

    def backward_pass(self, vector_index:int, difference_vector:ndarray) -> None:
        self.dictionary_vectors.update(
            gradient=-2 * difference_vector @ self.sparse_positive_vectors[vector_index].values.T + 2 * self.l2_regularisation_penalty * self.dictionary_vectors.values
        )
        self.sparse_positive_vectors[vector_index].update(
            gradient=-2 * self.dictionary_vectors.values.T @ difference_vector, 
        )

    def learn_sparse_vectors(self,dense_vectors:ndarray) -> None:
        average_error,previous_average_error = 1,0
        for iteration in range(self.n_iterations):
            if self.stopping_condition(
                iteration=iteration,
                average_error=average_error,
                average_error_delta=abs(average_error - previous_average_error)
            ):
                break
            total_error,l1_error = 0,0
            for index, word_vector in enumerate(dense_vectors):
                predicted_vector = self.forward_pass(vector_index=index)
                delta = word_vector - predicted_vector
                total_error += np_sum(delta ** 2)
                l1_error += np_sum(np_abs(self.sparse_positive_vectors[index].values)) 
                self.backward_pass(
                    vector_index=index, 
                    difference_vector=delta, 
                )
                print(f"\rProcessed words: {index}", end='')
            previous_average_error = average_error
            average_error = total_error / len(dense_vectors)
            print(f"\nIteration: {iteration}\nError per example: {average_error}\nDict L2 norm: {np_norm(self.dictionary_vectors.values)}\nAvg Atom L1 norm: {l1_error / len(dense_vectors)}")
    
    def stopping_condition(
        self, 
        iteration:int,
        average_error:float, 
        average_error_delta:float
    ) -> bool:
        return iteration > self.max_iterations and average_error > self.max_average_error and average_error_delta > self.max_average_error_delta

    def sparse_vectors(self) -> ndarray:
        return array([vector.values.flatten() for vector in self.sparse_positive_vectors])
    
