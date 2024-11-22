from numpy import array, concatenate

from predictors.model_based_predictor.transformer import Transformer
from predictors.model_based_predictor.train import train_model_with_target_embeddings
from predictors.model_based_predictor.predict import predict_n_encoded
from metrics.emergence import IntegratedInformation
from utils.encoder import eca_encoder, eca_decoder, eca_and_emergence_encoder
from utils.plotting import plot_results, plot_results_with_emergence
from utils.data_loader import generate_dataset


#--------Generate Train, Test Data----------

batch_index_to_display = 0
lattice_width=50
emergence_context_length = 7
transformer_context_length = 15
forecast_length=100
rule_number=3
ics = [
    682918332392260,
#    511854315302018,
#    635621643137219,
]
batch_size = len(ics)
n_epochs = 100
emergence_filter_context_history= 7

context_data, train_test_data = generate_dataset(
    rule_number=rule_number,
    lattice_width=lattice_width,
    batch_size=batch_size,
    context_sequence_length=transformer_context_length,
    max_sequence_length=forecast_length*2,
    initial_configurations=ics
) 

train_test_data_encoded = [
    [
        eca_encoder(
            index=index, 
            array_size=lattice_width,
        ) for index in sequence
     ] for sequence in train_test_data
]
train_data = array([
    sequence[:forecast_length]
    for sequence in train_test_data
])



#------Setup models--------

model_spacetime_only = Transformer(
    src_vocab_size= lattice_width, 
    tgt_vocab_size=lattice_width, 
    max_seq_length=forecast_length, 
    src_encoder=lambda indexes,array_size:eca_encoder(
        index=indexes[-1],
        array_size=array_size
    ),
    tgt_encoder=lambda indexes,array_size:eca_encoder(
        index=indexes[-1],
        array_size=array_size
    ),
)

model_spacetime_and_emergence = Transformer(
    src_vocab_size=lattice_width, 
    tgt_vocab_size=lattice_width, 
    max_seq_length=forecast_length, 
    src_encoder=lambda indexes,array_size:sum(eca_and_emergence_encoder(
        sequence=indexes,
        array_size=array_size,
        historical_context_length=emergence_context_length
    )),
    tgt_encoder=lambda indexes,array_size:eca_encoder(
        index=indexes[-1],
        array_size=array_size
    ),
)

#--------Train models----------

train_model_with_target_embeddings( 
   n_epochs=n_epochs,
   model=model_spacetime_only,
   x_train=context_data,
   y_train=train_data,
)

train_model_with_target_embeddings( 
   n_epochs=n_epochs,
   model=model_spacetime_and_emergence,
   x_train=context_data,
   y_train=train_data,
)



#------Predict with models--------


predicted_spacetime_only_a = predict_n_encoded(
    model=model_spacetime_only, 
    source=context_data,
    target=train_data[:,:1],
    batch_size=batch_size,
    forecast_horizon=forecast_length-1,
    vector_to_index=lambda vector: eca_decoder(
        lattice=vector,
        binary_threshold=0.0
    )
)
#TODO: for context use last 15 of the prev prediction - and then seed with final prediction
predicted_spacetime_only_b = predict_n_encoded(
    model=model_spacetime_only, 
    source=train_data[:,-emergence_context_length:],
    target=predicted_spacetime_only_a[:,:1],
    batch_size=batch_size,
    forecast_horizon=forecast_length-1,
    vector_to_index=lambda vector: eca_decoder(
        lattice=vector,
        binary_threshold=0.0
    )
)


predicted_spacetime_only_a_encoded = array([
    [
        eca_encoder(
            index=i,
            array_size=lattice_width
        ) for i in sequence
    ]
    for sequence in predicted_spacetime_only_a
])
predicted_spacetime_only_b_encoded = array([
    [
        eca_encoder(
            index=i,
            array_size=lattice_width
        ) for i in sequence
    ]
    for sequence in predicted_spacetime_only_b
])
predicted_spacetime_only_encoded = [
    concatenate([sequence_a, sequence_b])
    for sequence_a,sequence_b in zip(
        predicted_spacetime_only_a_encoded,
        predicted_spacetime_only_b_encoded
    )
]

predicted_spacetime_and_emergence_a = predict_n_encoded(
    model=model_spacetime_and_emergence, 
    source=context_data,
    target=train_data[:,:1],
    batch_size=batch_size,
    forecast_horizon=forecast_length-1,
    vector_to_index=lambda vector: eca_decoder(
        lattice=vector,
        binary_threshold=0.0
    )
)
predicted_spacetime_and_emergence_b = predict_n_encoded(
    model=model_spacetime_and_emergence, 
    source=train_data[:,-emergence_context_length:],
    target=predicted_spacetime_and_emergence_a[:,:1],
    batch_size=batch_size,
    forecast_horizon=forecast_length-1,
    vector_to_index=lambda vector: eca_decoder(
        lattice=vector,
        binary_threshold=0.0
    )
)
predicted_spacetime_and_emergence_a_encoded = array([
    [
        eca_encoder(
            index=i,
            array_size=lattice_width
        ) for i in sequence
    ]
    for sequence in predicted_spacetime_and_emergence_a
])
predicted_spacetime_and_emergence_b_encoded = array([
    [
        eca_encoder(
            index=i,
            array_size=lattice_width
        ) for i in sequence
    ]
    for sequence in predicted_spacetime_and_emergence_b
])
predicted_spacetime_and_emergence_encoded = [
    concatenate([sequence_a, sequence_b])
    for sequence_a,sequence_b in zip(
        predicted_spacetime_and_emergence_a_encoded,
        predicted_spacetime_and_emergence_b_encoded
    )
]


#------Display results--------

plot_results_with_emergence(
    title_text="Spacetime Only",
    real_spacetime_evolution=train_test_data_encoded[batch_index_to_display],
    predicted_spacetime_evolution=predicted_spacetime_only_encoded[batch_index_to_display],
    lattice_width=lattice_width,
    filter_spacetime_evolution=lambda spacetime:IntegratedInformation(k=emergence_context_length).emergence_filter(spacetime),
)

plot_results_with_emergence(
    title_text="Spacetime with Emergent Properties",
    real_spacetime_evolution=train_test_data_encoded[batch_index_to_display],
    predicted_spacetime_evolution=predicted_spacetime_and_emergence_encoded[batch_index_to_display],
    lattice_width=lattice_width,
    filter_spacetime_evolution=lambda spacetime:IntegratedInformation(k=emergence_context_length).emergence_filter(spacetime),
)