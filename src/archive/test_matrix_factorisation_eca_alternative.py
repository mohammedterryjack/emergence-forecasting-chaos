"""Experimentation with alternative decoding method 
to allow to predict any vector in configuration space
method doesnt work
"""

from typing import Generator
from numpy import array, ones, ndarray, zeros

from predictors.non_neural_predictors.matrix_factorisation import (
    matrix_factorisation_pseudo_inverse
)
from dynamical_system.eca.elementary_cellular_automata import ElementaryCellularAutomata
from utils_plotting import plot_spacetime_diagrams, plot_trajectories
from utils_projection import projector 



def construct_sparse_correlation_matrix_lattice_sized(lattice_evolution:list[list[int]], lattice_size:int) -> ndarray:
    sparse_matrix = zeros((lattice_size, lattice_size))
    current_vectors = lattice_evolution[:-1]
    next_vectors = lattice_evolution[1:]
    for current_vector,next_vector in zip(current_vectors, next_vectors):
        for current_index in range(lattice_size):
            for next_index in range(lattice_size):
                if current_vector[current_index] == next_vector[next_index]:
                    value = 0
                elif current_vector[current_index] > next_vector[next_index]:
                    value = -1
                elif current_vector[current_index] < next_vector[next_index]:
                    value = 1
                sparse_matrix[current_index,next_index] += value
    return sparse_matrix / len(lattice_evolution)



def predict_next_vector(x:ndarray, trained_embeddings:ndarray) -> ndarray:
    y = x @ trained_embeddings
    return y

def predict_n_vectors(seed_vector:list[int], n:int, trained_embeddings:ndarray, binary_threshold:float) -> Generator[ndarray,None,None]:
    vector = seed_vector
    for _ in range(n):
        vector_ = predict_next_vector(
            x=vector,
            trained_embeddings=trained_embeddings
        )
        vector = vector_ > binary_threshold
        yield vector



lattice_width=50
time_steps=100
rule_number=30

train_ca = ElementaryCellularAutomata(
    lattice_width=lattice_width,
    time_steps=time_steps,
    transition_rule_number=rule_number
)

matrix_mapping_current_lattice_to_next_lattice = construct_sparse_correlation_matrix_lattice_sized(
    lattice_evolution=train_ca.evolution,
    lattice_size=train_ca.info().lattice_width
)

current_vectors = ones((train_ca.info().lattice_width, train_ca.info().lattice_width))

next_vectors = matrix_factorisation_pseudo_inverse(
    sparse_matrix_to_factorise=matrix_mapping_current_lattice_to_next_lattice,
    factor_matrix_a=current_vectors,
)

predicted_vectors = list(predict_n_vectors(
    n=train_ca.info().time_steps,
    seed_vector=train_ca.evolution[0],
    trained_embeddings=next_vectors,
    binary_threshold=0.0
))

plot_trajectories(
    target=[[
        projector(
            embedding=vector,
            lattice_width=train_ca.info().lattice_width
        ) for vector in train_ca.evolution
    ]],
    predicted=[[
        projector(
            embedding=vector,
            lattice_width=train_ca.info().lattice_width
        ) for vector in predicted_vectors
    ]],
    batch_size=1
)

plot_spacetime_diagrams(
    target=[train_ca.evolution], 
    predicted=[predicted_vectors],
    batch_size=1
)

