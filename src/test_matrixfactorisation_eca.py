from numpy import array

from predictors.non_neural_predictors.matrix_factorisation import (
    matrix_factorisation_pseudo_inverse, 
    construct_memory_efficient_sparse_correlation_matrix,
    predict_n
)
from dynamical_system.eca.elementary_cellular_automata import ElementaryCellularAutomata
from utils_encoder import eca_encoder
from utils_projection import projector 
from utils_plotting import plot_spacetime_diagrams, plot_trajectories

lattice_width=50
time_steps=100
rule_number=3

train_ca = ElementaryCellularAutomata(
    lattice_width=lattice_width,
    time_steps=time_steps,
    transition_rule_number=rule_number
)

matrix_mapping_current_id_to_next_id,new_index_mapping = construct_memory_efficient_sparse_correlation_matrix(
    indexes=train_ca.info().lattice_evolution
)

current_vectors = array([
    eca_encoder(
        index=index, array_size=train_ca.info().lattice_width
    ) for index in new_index_mapping 
])

next_vectors = matrix_factorisation_pseudo_inverse(
    sparse_matrix_to_factorise=matrix_mapping_current_id_to_next_id,
    factor_matrix_a=current_vectors,
)

_, predicted_vectors = zip(*predict_n(
    n=train_ca.info().time_steps,
    seed_index=new_index_mapping.index(train_ca.info().lattice_evolution[0]),
    trained_embeddings=next_vectors,
    index_to_vector=lambda index: eca_encoder(
        index=new_index_mapping[index], 
        array_size=train_ca.info().lattice_width
    )
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


test_ca = ElementaryCellularAutomata(
    initial_state=train_ca.info().lattice_evolution[-1],
    lattice_width=lattice_width,
    time_steps=time_steps,
    transition_rule_number=rule_number
)

_, test_predicted_vectors = zip(*predict_n(
    n=test_ca.info().time_steps,
    seed_index=new_index_mapping.index(
        train_ca.info().lattice_evolution[-1]
    ),
    trained_embeddings=next_vectors,
    index_to_vector=lambda index: eca_encoder(
        index=new_index_mapping[index], 
        array_size=test_ca.info().lattice_width
    )
))

plot_trajectories(
    target=[[
        projector(
            embedding=vector,
            lattice_width=test_ca.info().lattice_width
        ) for vector in test_ca.evolution
    ]],
    predicted=[[
        projector(
            embedding=vector,
            lattice_width=test_ca.info().lattice_width
        ) for vector in test_predicted_vectors
    ]],
    batch_size=1
)

plot_spacetime_diagrams(
    target=[test_ca.evolution], 
    predicted=[test_predicted_vectors],
    batch_size=1
)
