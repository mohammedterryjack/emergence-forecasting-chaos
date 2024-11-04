from typing import Generator 

from numpy import ndarray, isnan, zeros
from numpy.random import rand
from numpy.linalg import pinv 
from matplotlib.pyplot import plot, show 

def matrix_factorisation_pseudo_inverse(factor_matrix_a:ndarray, sparse_matrix_to_factorise:ndarray) -> ndarray:    
    """Factorise the sparse matrix into 2 smaller factor matrices A and B, 
    where factor-matrix A is known
    Pseudo Inverse is used to find factor matrix B"""
    return pinv(factor_matrix_a) @ sparse_matrix_to_factorise   

def matrix_factorisation_sgd(
    factor_matrix_a: ndarray, sparse_matrix_to_factorise: ndarray,
    n_iterations:int=1000, learning_rate=1e-3, verbose:bool=True
) -> ndarray:
    """Factorise the sparse matrix into 2 smaller factor matrices A and B, 
    where factor-matrix A is known
    Stochastic Gradient Descent is used to find factor matrix B"""

    num_rows, num_cols = sparse_matrix_to_factorise.shape
    _,num_factors = factor_matrix_a.shape 

    errors = []
    factor_matrix_b = rand(num_factors, num_cols) 

    for iteration in range(n_iterations):
        total_error = 0.0
        for i in range(num_rows):
            for j in range(num_cols):
                r_ui = sparse_matrix_to_factorise[i, j]
                if isnan(r_ui): 
                    continue
                pred_r_ui = factor_matrix_a[i, :] @ factor_matrix_b[:, j]
                e_ui = r_ui - pred_r_ui
                factor_matrix_b[:, j] = factor_matrix_b[:, j] + learning_rate * (
                    e_ui * factor_matrix_a[i, :] - factor_matrix_b[:, j]
                )
                total_error += abs(e_ui)
        if verbose:
            print(f"iteration {iteration}: loss={total_error}")
        errors.append(total_error)
    if verbose:
        plot(errors)
        show()
    return factor_matrix_b

def construct_sparse_correlation_matrix(indexes:list[int], vector_size:int) -> ndarray:
    sparse_matrix = zeros((vector_size, vector_size))
    current_indices = indexes[:-1]
    next_indices = indexes[1:]
    sparse_matrix[current_indices, next_indices] = 1
    return sparse_matrix

def construct_memory_efficient_sparse_correlation_matrix(indexes:list[int]) -> tuple[ndarray, list[int]]:
    """only creates sparse matrix contianing indices visited in the trajectory 
    - not all possible indices in the configuration space
    which means only the configurations seen can ever be predicted
    this is similar to a lookup table using the vectors instead of the indexes
    which means we can combine the emergent features (as vectors) into the lookup
    """
    original_to_mini_index_mapping = list(set(indexes))
    original_to_mini_index_mapping.sort()
    new_indexes = [original_to_mini_index_mapping.index(i) for i in indexes]

    current_indices = new_indexes[:-1]
    next_indices = new_indexes[1:]

    vector_size = len(original_to_mini_index_mapping)
    sparse_matrix = zeros((vector_size, vector_size))
    sparse_matrix[current_indices, next_indices] = 1

    return sparse_matrix, original_to_mini_index_mapping


def predict_next(x:ndarray, trained_embeddings:ndarray) -> ndarray:
    y = x @ trained_embeddings
    return y.argmax()

def predict_n(seed_index:int, n:int, index_to_vector:callable, trained_embeddings:ndarray) -> Generator[tuple[int,ndarray],None,None]:
    index = seed_index   
    for _ in range(n):
        vector = index_to_vector(index=index)
        index = predict_next(
            x=vector,
            trained_embeddings=trained_embeddings
        )
        yield index, vector
