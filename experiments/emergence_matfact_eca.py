from numpy import array

from predictors.model_free_predictor.matrix_factorisation import (
    matrix_factorisation_pseudo_inverse, 
    construct_memory_efficient_sparse_correlation_matrix,
    predict_n
)
from utils.encoder import eca_encoder, EncoderOption
from utils.plotting import plot_results 
from utils.data_loader import generate_dataset


lattice_width=50
forecast_length=70
rule_number=3
ics = [
    682918332392260,
    511854315302018,
    635621643137219,
]
batch_size = len(ics)

_, target_data = generate_dataset(
    rule_number=rule_number,
    lattice_width=lattice_width,
    batch_size=batch_size,
    context_sequence_length=0,
    max_sequence_length=forecast_length,
    initial_configurations=ics
) 

matrix_mapping_current_id_to_next_id,new_index_mapping = construct_memory_efficient_sparse_correlation_matrix(
    indexes=array(target_data).reshape(-1)
)

encoder_option = EncoderOption.SPACETIME_AND_EMERGENCE

current_vectors = array([
    eca_encoder(
        index=index, 
        array_size=lattice_width,
        option=encoder_option
    ) for index in new_index_mapping 
])

next_vectors = matrix_factorisation_pseudo_inverse(
    sparse_matrix_to_factorise=matrix_mapping_current_id_to_next_id,
    factor_matrix_a=current_vectors,
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
            option=encoder_option
        )
    ))
    test_predicted_data.append(test_predicted_indexes)

test_predicted_data_vectors = [
    [
        eca_encoder(
            index=new_index_mapping[index], 
            array_size=lattice_width,
            option=EncoderOption.SPACETIME_ONLY
        ) for index in test_predicted_data[b]
     ] for b in range(batch_size)
]

test_target_data_vectors = [
    [
        eca_encoder(
            index=index, 
            array_size=lattice_width,
            option=EncoderOption.SPACETIME_ONLY
        ) for index in test_target_data[b]
     ] for b in range(batch_size)
]

plot_results(
    target=test_target_data_vectors,
    predicted=test_predicted_data_vectors,
    batch_size=batch_size,
    lattice_width=lattice_width,
    timesteps=list(range(forecast_length))
)
