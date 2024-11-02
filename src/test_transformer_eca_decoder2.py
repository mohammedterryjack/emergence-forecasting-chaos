from numpy import array

from predictors.neural_predictor.transformer import Transformer
from predictors.neural_predictor.train import train_model_with_target_embeddings
from predictors.neural_predictor.predict import predict_n_encoded
from utils_projection import projector
from utils_encoder import eca_encoder, eca_decoder
from utils_plotting import plot_trajectories, plot_spacetime_diagrams
from utils_data_loader import generate_dataset


rule_number=30
lattice_width = 50
context_length = 2
forecast_length = 50
batch_size = 3
n_epochs = 100

source_data, target_data_, new_index_mapping = generate_dataset(
    rule_number=rule_number,
    lattice_width=lattice_width,
    batch_size=batch_size,
    context_sequence_length=context_length,
    max_sequence_length=forecast_length
) 
target_data = array([
    [
        new_index_mapping[index] for index in target_data_[b]
    ]
    for b in range(batch_size)
])


model = Transformer(
    src_vocab_size=lattice_width, 
    tgt_vocab_size=lattice_width, 
    max_seq_length=forecast_length, 
    src_encoder=eca_encoder,
    tgt_encoder=eca_encoder
)

train_model_with_target_embeddings( 
   n_epochs=n_epochs,
   model=model,
   x_train=source_data,
   y_train=target_data,
)

predicted_data = predict_n_encoded(
    model=model, 
    source=source_data,
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
            array_size=lattice_width
        ) for i in target_data[b]
    ]
    for b in range(batch_size)
]

predicted_data_encoded = array([
    [
        model.encoder_embedding.index_encoder(
            index=i,
            array_size=model.encoder_embedding.vocab_size
        ) for i in predicted_data[b]
    ]
    for b in range(batch_size)
])

plot_trajectories(
    target=[
        [
            projector(
                embedding=embedding,
                lattice_width=lattice_width
            ) for embedding in target_data_encoded[b]
        ]
        for b in range(batch_size)
    ], 
    predicted=[
        [
            projector(
                embedding=embedding,
                lattice_width=lattice_width
            ) for embedding in predicted_data_encoded[b]
        ]
        for b in range(batch_size)
    ],
    batch_size=batch_size
)

plot_spacetime_diagrams(
    target=target_data_encoded,
    predicted=predicted_data_encoded,
    batch_size=batch_size
)

#TODO:
# - refactor all to use new utils etc
# - test using newly generated trajectories (not train ones)
# - try predicting by adding additional / emergent features in src_encoder
