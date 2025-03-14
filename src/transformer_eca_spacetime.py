"""Using new decoder method 
- binary thresholding vector 
so can also predict new configurations 
which were never seen during training
"""

from numpy import array

from predictors.model_based_predictor.transformer import Transformer
from predictors.model_based_predictor.train import train_model_with_target_embeddings
from predictors.model_based_predictor.predict import predict_n_encoded
from utils.encoder import eca_encoder, eca_decoder
from utils.plotting import plot_results 
from utils.data_loader import generate_dataset


lattice_width = 50
forecast_length = 70
rule_number=30
ics = [
    #1,
    682918332392260,
    511854315302018,
    635621643137219,
    #26398899248128
]
batch_size = len(ics)
n_epochs = 100
context_length = 10

context_data, target_data = generate_dataset(
    rule_number=rule_number,
    lattice_width=lattice_width,
    batch_size=batch_size,
    context_sequence_length=context_length,
    max_sequence_length=forecast_length,
    initial_configurations=ics
) 
target_data = array(target_data)

model = Transformer(
    src_vocab_size= lattice_width, 
    tgt_vocab_size=lattice_width, 
    max_seq_length=forecast_length, 
    #src_encoder=eca_encoder,
    #tgt_encoder=eca_encoder,
    src_encoder=lambda indexes,array_size:eca_encoder(
        index=indexes[-1],
        array_size=array_size
    ),
    tgt_encoder=lambda indexes,array_size:eca_encoder(
        index=indexes[-1],
        array_size=array_size
    ),
)

train_model_with_target_embeddings( 
   n_epochs=n_epochs,
   model=model,
   x_train=context_data,
   y_train=target_data,
)


predicted_data = predict_n_encoded(
    model=model, 
    source=context_data,
    target=target_data[:,:1],
    batch_size=batch_size,
    forecast_horizon=forecast_length-1,
    vector_to_index=lambda vector: eca_decoder(
        lattice=vector,
        binary_threshold=0.0
    )
)

target_data_encoded=[
    [
        eca_encoder(
            index=i,
            array_size=lattice_width,
        ) for i in target_data[b]
    ]
    for b in range(batch_size)
]

predicted_data_encoded = array([
   [
       model.encoder_embedding.index_encoder(
           indexes=[i],
           array_size=model.encoder_embedding.vocab_size
       ) for i in predicted_data[b]
   ]
   for b in range(batch_size)
])


plot_results(
    target=target_data_encoded,
    predicted=predicted_data_encoded,
    batch_size=batch_size,
    lattice_width=lattice_width,
    timesteps=list(range(forecast_length))
)


test_context_data, test_target_data = generate_dataset(
    rule_number=rule_number,
    lattice_width=lattice_width,
    batch_size=batch_size,
    context_sequence_length=context_length,
    max_sequence_length=forecast_length
) 
test_target_data = array(test_target_data)

test_predicted_data = predict_n_encoded(
    model=model, 
    source=test_context_data,
    target=test_target_data[:,:1],
    batch_size=batch_size,
    forecast_horizon=forecast_length-1,
    vector_to_index=lambda vector: eca_decoder(
        lattice=vector,
        binary_threshold=0.0
    )
)

test_target_data_encoded=[
    [
        eca_encoder(
            index=i,
            array_size=lattice_width,
        ) for i in test_target_data[b]
    ]
    for b in range(batch_size)
]

test_predicted_data_encoded = array([
    [
        model.encoder_embedding.index_encoder(
            indexes=[i],
            array_size=model.encoder_embedding.vocab_size
        ) for i in test_predicted_data[b]
    ]
    for b in range(batch_size)
])



plot_results(
    target=test_target_data_encoded,
    predicted=test_predicted_data_encoded,
    batch_size=batch_size,
    lattice_width=lattice_width,
    timesteps=list(range(forecast_length))
)
