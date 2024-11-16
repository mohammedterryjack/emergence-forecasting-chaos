from numpy import array, concatenate, where

from predictors.model_free_predictor.matrix_factorisation import (
    matrix_factorisation_pseudo_inverse, 
    construct_memory_efficient_sparse_correlation_matrix,
    predict_n
)
from metrics.emergence import IntegratedInformation
from utils.encoder import eca_encoder, eca_and_emergence_encoder
from utils.plotting import plot_results, plot_results_with_emergence
from utils.data_loader import generate_dataset


#--------Generate Train, Test Data----------

emergence_context_length = 7
lattice_width=50
forecast_length=100
rule_number=3
ics = [
    #682918332392260,
    511854315302018,
    #635621643137219,
]
batch_size = len(ics)
batch_index_to_display = 0

train_data, test_data = generate_dataset(
    rule_number=rule_number,
    lattice_width=lattice_width,
    batch_size=batch_size,
    context_sequence_length=forecast_length,
    max_sequence_length=forecast_length,
    initial_configurations=ics
) 

train_test_data_encoded = [
    [
        eca_encoder(
            index=index, 
            array_size=lattice_width,
        ) for index in train_sequence + test_sequence
     ] for train_sequence,test_sequence in zip(train_data,test_data)
]

#------Setup models--------
matrix_mapping_current_id_to_next_id,new_index_mapping = construct_memory_efficient_sparse_correlation_matrix(
    indexes=array(train_data).reshape(-1)
)

new_indexes_with_context = []
for index in new_index_mapping:
    for b,i in zip(*where(train_data==index)):
        start = max(0,i-emergence_context_length)
        end = i+1
        new_indexes_with_context.append(train_data[b][start:end])
        break

current_vectors_encoded_without_emergence = array([
    eca_encoder(
        index=index, 
        array_size=lattice_width,
    ) for index in new_index_mapping 
])
current_vectors_encoded_only_emergence = array([
    eca_and_emergence_encoder(
        sequence=context,
        array_size=lattice_width,
        historical_context_length=emergence_context_length
    )[-1] for context in new_indexes_with_context
])
current_vectors_encoded_with_emergence = array([
    sum(eca_and_emergence_encoder(
    #concatenate(eca_and_emergence_encoder(
        sequence=context,
        array_size=lattice_width,
        historical_context_length=emergence_context_length
    )) for context in new_indexes_with_context
])

#--------Train models----------

next_vectors_encoded_without_emergence = matrix_factorisation_pseudo_inverse(
    sparse_matrix_to_factorise=matrix_mapping_current_id_to_next_id,
    factor_matrix_a=current_vectors_encoded_without_emergence,
)
next_vectors_encoded_only_emergence = matrix_factorisation_pseudo_inverse(
   sparse_matrix_to_factorise=matrix_mapping_current_id_to_next_id,
   factor_matrix_a=current_vectors_encoded_only_emergence,
)
next_vectors_encoded_with_emergence = matrix_factorisation_pseudo_inverse(
   sparse_matrix_to_factorise=matrix_mapping_current_id_to_next_id,
   factor_matrix_a=current_vectors_encoded_with_emergence,
)

#------Predict with models--------


predicted_data_without_emergence = []
for ic in ics:
    predicted_indexes, _ = zip(*predict_n(
        n=forecast_length*2,
        seed_index=new_index_mapping.index(ic),
        trained_embeddings=next_vectors_encoded_without_emergence,
        index_sequence_to_vector=lambda sequence: eca_encoder(
            index=new_index_mapping[sequence[-1]], 
            array_size=lattice_width,
        )
    ))
    predicted_vectors = [
        eca_encoder(
            index=new_index_mapping[index], 
            array_size=lattice_width,
        ) for index in predicted_indexes
     ]
    predicted_data_without_emergence.append(predicted_vectors)

predicted_data_with_emergence = []
for ic in ics:
    predicted_indexes, _ = zip(*predict_n(
        n=forecast_length*2,
        seed_index=new_index_mapping.index(ic),
        trained_embeddings=next_vectors_encoded_with_emergence,
        index_sequence_to_vector=lambda sequence: sum(eca_and_emergence_encoder(
        #concatenate(eca_and_emergence_encoder(
            sequence=[new_index_mapping[i] for i in sequence],
            array_size=lattice_width,
            historical_context_length=emergence_context_length
        ))       
    ))

    predicted_vectors = [
       eca_encoder(
           index=new_index_mapping[index], 
           array_size=lattice_width,
       ) for index in predicted_indexes
    ]
    predicted_data_with_emergence.append(predicted_vectors)

predicted_data_only_emergence = []
for ic in ics:
    predicted_indexes, _ = zip(*predict_n(
        n=forecast_length*2,
        seed_index=new_index_mapping.index(ic),
        trained_embeddings=next_vectors_encoded_only_emergence,
        index_sequence_to_vector=lambda sequence: eca_and_emergence_encoder(
            sequence=[new_index_mapping[i] for i in sequence],
            array_size=lattice_width,
            historical_context_length=emergence_context_length
        )[-1],
    ))
    predicted_vectors = [
        eca_encoder(
            index=new_index_mapping[index], 
            array_size=lattice_width,
        ) for index in predicted_indexes
     ]
    predicted_data_only_emergence.append(predicted_vectors)

#------Display results--------


plot_results_with_emergence(
    title_text="Spacetime Only",
    real_spacetime_evolution=train_test_data_encoded[batch_index_to_display],
    predicted_spacetime_evolution=predicted_data_without_emergence[batch_index_to_display],
    lattice_width=lattice_width,
    filter_spacetime_evolution=lambda spacetime:IntegratedInformation(k=emergence_context_length).emergence_filter(spacetime),
)
plot_results_with_emergence(
    title_text="Spacetime and Emergent Properties",
    real_spacetime_evolution=train_test_data_encoded[batch_index_to_display],
    predicted_spacetime_evolution=predicted_data_with_emergence[batch_index_to_display],
    lattice_width=lattice_width,
    filter_spacetime_evolution=lambda spacetime:IntegratedInformation(k=emergence_context_length).emergence_filter(spacetime),
)
plot_results_with_emergence(
    title_text="Emergent Properties Only",
    real_spacetime_evolution=train_test_data_encoded[batch_index_to_display],
    predicted_spacetime_evolution=predicted_data_only_emergence[batch_index_to_display],
    lattice_width=lattice_width,
    filter_spacetime_evolution=lambda spacetime:IntegratedInformation(k=emergence_context_length).emergence_filter(spacetime),
)
