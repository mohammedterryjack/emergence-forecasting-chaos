from numpy import array

from predictors.model_free_predictor.matrix_factorisation import (
    matrix_factorisation_pseudo_inverse, 
    construct_memory_efficient_sparse_correlation_matrix,
    predict_n
)
from utils.encoder import eca_encoder
from utils.plotting import plot_results 
from utils.data_loader import generate_dataset

batch_size = 2
lattice_width=50
forecast_length=50
rule_number=3
include_emergent_features = True


_, target_data = generate_dataset(
    rule_number=rule_number,
    lattice_width=lattice_width,
    batch_size=batch_size,
    context_sequence_length=0,
    max_sequence_length=forecast_length
) 


indexes = []
for b in range(batch_size):    
    indexes.extend(target_data[b])

matrix_mapping_current_id_to_next_id,new_index_mapping = construct_memory_efficient_sparse_correlation_matrix(
    indexes=indexes
)

current_vectors = array([
    eca_encoder(
        index=index, 
        array_size=lattice_width,
        include_emergent_features=include_emergent_features
    ) for index in new_index_mapping 
])

next_vectors = matrix_factorisation_pseudo_inverse(
    sparse_matrix_to_factorise=matrix_mapping_current_id_to_next_id,
    factor_matrix_a=current_vectors,
)

predicted_data = []
for b in range(batch_size):
    predicted_indexes, _ = zip(*predict_n(
        n=forecast_length,
        seed_index=new_index_mapping.index(target_data[b][0]),
        trained_embeddings=next_vectors,
        index_to_vector=lambda index: eca_encoder(
            index=new_index_mapping[index], 
            array_size=lattice_width,
            include_emergent_features=include_emergent_features
        )
    ))
    predicted_data.append(predicted_indexes)

predicted_data_vectors = [
    [
        eca_encoder(
            index=new_index_mapping[index], 
            array_size=lattice_width,
        ) for index in predicted_data[b]
     ] for b in range(batch_size)
]

target_data_vectors = [
    [
        eca_encoder(
            index=index, 
            array_size=lattice_width,
        ) for index in target_data[b]
     ] for b in range(batch_size)
]

plot_results(
    target=target_data_vectors,
    predicted=predicted_data_vectors,
    batch_size=batch_size,
    lattice_width=lattice_width
)


initial_configurations = [
    target_data[b][-1]
    for b in range(batch_size)
]

_, test_target_data = generate_dataset(
    rule_number=rule_number,
    lattice_width=lattice_width,
    batch_size=batch_size,
    context_sequence_length=0,
    max_sequence_length=forecast_length,
    initial_configurations=initial_configurations
) 

test_predicted_data = []
for b in range(batch_size):
    ic = initial_configurations[b]
    test_predicted_indexes, _ = zip(*predict_n(
        n=forecast_length,
        seed_index=new_index_mapping.index(ic),
        trained_embeddings=next_vectors,
        index_to_vector=lambda index: eca_encoder(
            index=new_index_mapping[index], 
            array_size=lattice_width,
            include_emergent_features=include_emergent_features
        )
    ))
    test_predicted_data.append(test_predicted_indexes)

test_predicted_data_vectors = [
    [
        eca_encoder(
            index=new_index_mapping[index], 
            array_size=lattice_width,
        ) for index in test_predicted_data[b]
     ] for b in range(batch_size)
]

test_target_data_vectors = [
    [
        eca_encoder(
            index=index, array_size=lattice_width
        ) for index in test_target_data[b]
     ] for b in range(batch_size)
]


plot_results(
    target=test_target_data_vectors,
    predicted=test_predicted_data_vectors,
    batch_size=batch_size,
    lattice_width=lattice_width
)
