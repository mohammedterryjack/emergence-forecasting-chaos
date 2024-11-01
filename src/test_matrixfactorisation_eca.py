from typing import Generator 

from numpy import array, ndarray

from predictors.non_neural_predictors.matrix_factorisation import matrix_factorisation_pseudo_inverse, construct_memory_efficient_sparse_correlation_matrix
from dynamical_system.eca.elementary_cellular_automata import ElementaryCellularAutomata
from utils_encoder import eca_encoder
from utils_projection import projector 
from utils_plotting import plot_spacetime_diagrams, plot_trajectories

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


ca = ElementaryCellularAutomata(
    lattice_width=50,
    time_steps=100,
    transition_rule_number=30
)
metadata = ca.info()

matrix_mapping_current_id_to_next_id,new_index_mapping = construct_memory_efficient_sparse_correlation_matrix(
    indexes=metadata.lattice_evolution[:len(metadata.lattice_evolution)//2]
)

current_vectors = array([
    eca_encoder(
        index=index, array_size=metadata.lattice_width
    ) for index in new_index_mapping #range(metadata.lattice_configuration_space)
])

next_vectors = matrix_factorisation_pseudo_inverse(
    sparse_matrix_to_factorise=matrix_mapping_current_id_to_next_id,
    factor_matrix_a=current_vectors,
)

_, predicted_vectors = zip(*predict_n(
    n=metadata.time_steps,
    seed_index=new_index_mapping.index(metadata.lattice_evolution[0]),
    trained_embeddings=next_vectors,
    index_to_vector=lambda index: eca_encoder(
        index=new_index_mapping[index], 
        array_size=metadata.lattice_width
    )
))

plot_spacetime_diagrams(
    target=[ca.evolution], 
    predicted=[predicted_vectors],
    batch_size=1
)
plot_trajectories(
    target=[[
        projector(
            embedding=vector,
            lattice_width=metadata.lattice_width
        ) for vector in ca.evolution
    ]],
    predicted=[[
        projector(
            embedding=vector,
            lattice_width=metadata.lattice_width
        ) for vector in predicted_vectors
    ]],
    batch_size=1
)


#TODO:
# - try predicting by adding additional / emergent features in src_encoder

