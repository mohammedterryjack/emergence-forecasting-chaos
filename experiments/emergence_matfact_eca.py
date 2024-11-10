from numpy import array

from predictors.model_free_predictor.matrix_factorisation import (
    matrix_factorisation_pseudo_inverse, 
    construct_memory_efficient_sparse_correlation_matrix,
    predict_n
)
from utils.encoder import eca_encoder, EncoderOption
from utils.plotting import plot_results, plot_results_with_emergence
from utils.data_loader import generate_dataset


#--------Generate Train, Test Data----------

lattice_width=50
forecast_length=100
rule_number=3
ics = [
    682918332392260,
#    511854315302018,
#    635621643137219,
]
batch_size = len(ics)

_, train_data = generate_dataset(
    rule_number=rule_number,
    lattice_width=lattice_width,
    batch_size=batch_size,
    context_sequence_length=0,
    max_sequence_length=forecast_length,
    initial_configurations=ics
) 

test_ics = [sequence[-1] for sequence in train_data]
_, test_data = generate_dataset(
    rule_number=rule_number,
    lattice_width=lattice_width,
    batch_size=batch_size,
    context_sequence_length=0,
    max_sequence_length=forecast_length,
    initial_configurations=test_ics
) 

test_data_encoded = [
    [
        eca_encoder(
            index=index, 
            array_size=lattice_width,
            option=EncoderOption.SPACETIME_ONLY
        ) for index in sequence
     ] for sequence in test_data
]

#------Setup models--------
matrix_mapping_current_id_to_next_id,new_index_mapping = construct_memory_efficient_sparse_correlation_matrix(
    indexes=array(train_data).reshape(-1)
)
current_vectors_encoded_without_emergence = array([
    eca_encoder(
        index=index, 
        array_size=lattice_width,
        option=EncoderOption.SPACETIME_ONLY
    ) for index in new_index_mapping 
])
current_vectors_encoded_with_emergence = array([
    eca_encoder(
        index=index, 
        array_size=lattice_width,
        option=EncoderOption.SPACETIME_AND_EMERGENCE
    ) for index in new_index_mapping 
])
current_vectors_encoded_only_emergence = array([
    eca_encoder(
        index=index, 
        array_size=lattice_width,
        option=EncoderOption.EMERGENCE_ONLY
    ) for index in new_index_mapping 
])

#--------Train models----------

next_vectors_encoded_without_emergence = matrix_factorisation_pseudo_inverse(
    sparse_matrix_to_factorise=matrix_mapping_current_id_to_next_id,
    factor_matrix_a=current_vectors_encoded_without_emergence,
)
next_vectors_encoded_with_emergence = matrix_factorisation_pseudo_inverse(
    sparse_matrix_to_factorise=matrix_mapping_current_id_to_next_id,
    factor_matrix_a=current_vectors_encoded_with_emergence,
)
next_vectors_encoded_only_emergence = matrix_factorisation_pseudo_inverse(
    sparse_matrix_to_factorise=matrix_mapping_current_id_to_next_id,
    factor_matrix_a=current_vectors_encoded_only_emergence,
)

#------Predict with models--------


encoder_option=EncoderOption.SPACETIME_ONLY
predicted_data_without_emergence = []
for test_ic in test_ics:
    predicted_indexes, _ = zip(*predict_n(
        n=forecast_length,
        seed_index=new_index_mapping.index(test_ic),
        trained_embeddings=next_vectors_encoded_without_emergence,
        index_to_vector=lambda index: eca_encoder(
            index=new_index_mapping[index], 
            array_size=lattice_width,
            option=encoder_option
        )
    ))
    predicted_vectors = [
        eca_encoder(
            index=new_index_mapping[index], 
            array_size=lattice_width,
            option=encoder_option
        ) for index in predicted_indexes
     ]
    predicted_data_without_emergence.append(predicted_vectors)

encoder_option=EncoderOption.SPACETIME_AND_EMERGENCE
predicted_data_with_emergence = []
for test_ic in test_ics:
    predicted_indexes, _ = zip(*predict_n(
        n=forecast_length,
        seed_index=new_index_mapping.index(test_ic),
        trained_embeddings=next_vectors_encoded_with_emergence,
        index_to_vector=lambda index: eca_encoder(
            index=new_index_mapping[index], 
            array_size=lattice_width,
            option=encoder_option
        )
    ))
    predicted_vectors = [
        eca_encoder(
            index=new_index_mapping[index], 
            array_size=lattice_width,
            option=encoder_option
        ) for index in predicted_indexes
     ]
    predicted_data_with_emergence.append(predicted_vectors)


encoder_option=EncoderOption.EMERGENCE_ONLY
predicted_data_only_emergence = []
for test_ic in test_ics:
    predicted_indexes, _ = zip(*predict_n(
        n=forecast_length,
        seed_index=new_index_mapping.index(test_ic),
        trained_embeddings=next_vectors_encoded_only_emergence,
        index_to_vector=lambda index: eca_encoder(
            index=new_index_mapping[index], 
            array_size=lattice_width,
            option=encoder_option
        )
    ))
    predicted_vectors = [
        eca_encoder(
            index=new_index_mapping[index], 
            array_size=lattice_width,
            option=encoder_option
        ) for index in predicted_indexes
     ]
    predicted_data_only_emergence.append(predicted_vectors)

#------Display results--------

plot_results_with_emergence(
    real=test_data_encoded[0],
    predicted=predicted_data_without_emergence[0],
    emergence_filter=lambda vector:vector,
    lattice_width=lattice_width,
)
plot_results_with_emergence(
    real=test_data_encoded[0],
    predicted=predicted_data_with_emergence[0],
    emergence_filter=lambda vector:vector,
    lattice_width=lattice_width,
)
plot_results_with_emergence(
    real=test_data_encoded[0],
    predicted=predicted_data_only_emergence[0],
    emergence_filter=lambda vector:vector,
    lattice_width=lattice_width,
)
