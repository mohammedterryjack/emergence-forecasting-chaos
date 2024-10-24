from numpy import ndarray, isnan
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
