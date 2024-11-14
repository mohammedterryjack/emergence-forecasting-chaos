from numpy import array

from predictors.model_based_predictor.transformer import Transformer
from predictors.model_based_predictor.train import train_model_with_target_embeddings
from predictors.model_based_predictor.predict import predict_n_encoded
from utils.encoder import eca_encoder, eca_decoder
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
n_epochs = 100

train_context_data, train_data = generate_dataset(
    rule_number=rule_number,
    lattice_width=lattice_width,
    batch_size=batch_size,
    context_sequence_length=0,
    max_sequence_length=forecast_length,
    initial_configurations=ics
) 
train_data = array(train_data)

test_ics = [sequence[-1] for sequence in train_data]
test_context_data, test_data = generate_dataset(
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
        ) for index in sequence
     ] for sequence in test_data
]

#------Setup models--------

model_spacetime_only = Transformer(
    src_vocab_size= lattice_width, 
    tgt_vocab_size=lattice_width, 
    max_seq_length=forecast_length, 
    src_encoder=lambda index,array_size:eca_encoder(
        index=index,
        array_size=array_size,
    ),
    tgt_encoder=lambda index,array_size:eca_encoder(
        index=index,
        array_size=array_size,
    )
)
model_spacetime_and_emergence = Transformer(
    src_vocab_size= lattice_width, 
    tgt_vocab_size=lattice_width, 
    max_seq_length=forecast_length, 
    src_encoder=lambda index,array_size:eca_encoder(
        index=index,
        array_size=array_size,
        #TODO: option=EncoderOption.SPACETIME_AND_EMERGENCE
    ),
    tgt_encoder=lambda index,array_size:eca_encoder(
        index=index,
        array_size=array_size,
    )
)
model_emergence_only = Transformer(
    src_vocab_size= lattice_width, 
    tgt_vocab_size=lattice_width, 
    max_seq_length=forecast_length, 
    src_encoder=lambda index,array_size:eca_encoder(
        index=index,
        array_size=array_size,
        #TODO: option=EncoderOption.EMERGENCE_ONLY
    ),
    tgt_encoder=lambda index,array_size:eca_encoder(
        index=index,
        array_size=array_size,
    )
)
#--------Train models----------

train_model_with_target_embeddings( 
   n_epochs=n_epochs,
   model=model_spacetime_only,
   x_train=train_context_data,
   y_train=train_data,
)

train_model_with_target_embeddings( 
   n_epochs=n_epochs,
   model=model_spacetime_and_emergence,
   x_train=train_context_data,
   y_train=train_data,
)

train_model_with_target_embeddings( 
   n_epochs=n_epochs,
   model=model_emergence_only,
   x_train=train_context_data,
   y_train=train_data,
)


#------Predict with models--------


predicted_spacetime_only = predict_n_encoded(
    model=model_spacetime_only, 
    source=test_context_data,
    target=train_data[:,:1],
    batch_size=batch_size,
    forecast_horizon=forecast_length-1,
    vector_to_index=lambda vector: eca_decoder(
        lattice=vector,
        binary_threshold=0.0
    )
)
predicted_spacetime_only_encoded = array([
    [
        model_spacetime_only.encoder_embedding.index_encoder(
            index=i,
            array_size=model_spacetime_only.encoder_embedding.vocab_size
        ) for i in sequence
    ]
    for sequence in predicted_spacetime_only
])



predicted_spacetime_and_emergence = predict_n_encoded(
    model=model_spacetime_and_emergence, 
    source=test_context_data,
    target=train_data[:,:1],
    batch_size=batch_size,
    forecast_horizon=forecast_length-1,
    vector_to_index=lambda vector: eca_decoder(
        lattice=vector,
        binary_threshold=0.0
    )
)
predicted_spacetime_and_emergence_encoded = array([
    [
        model_spacetime_and_emergence.encoder_embedding.index_encoder(
            index=i,
            array_size=model_spacetime_and_emergence.encoder_embedding.vocab_size
        ) for i in sequence
    ]
    for sequence in predicted_spacetime_and_emergence
])



predicted_emergence_only = predict_n_encoded(
    model=model_emergence_only, 
    source=test_context_data,
    target=train_data[:,:1],
    batch_size=batch_size,
    forecast_horizon=forecast_length-1,
    vector_to_index=lambda vector: eca_decoder(
        lattice=vector,
        binary_threshold=0.0
    )
)
predicted_emergence_only_encoded = array([
    [
        model_emergence_only.encoder_embedding.index_encoder(
            index=i,
            array_size=model_emergence_only.encoder_embedding.vocab_size
        ) for i in sequence
    ]
    for sequence in predicted_emergence_only
])


#------Display results--------

plot_results_with_emergence(
    real=test_data_encoded[0],
    predicted=predicted_spacetime_only_encoded[0],
    emergence_filter=lambda vector:vector,
    lattice_width=lattice_width,
)
plot_results_with_emergence(
    real=test_data_encoded[0],
    predicted=predicted_spacetime_and_emergence_encoded[0],
    emergence_filter=lambda vector:vector,
    lattice_width=lattice_width,
)
plot_results_with_emergence(
    real=test_data_encoded[0],
    predicted=predicted_emergence_only_encoded[0],
    emergence_filter=lambda vector:vector,
    lattice_width=lattice_width,
)