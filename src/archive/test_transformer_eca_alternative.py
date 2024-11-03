"""Using same decoder method as matrix factorisation 
(limitation: cannot predict new configurations outside of those seen during training
only can predict new configuration sequences)
"""

from numpy import array

from predictors.neural_predictor.transformer import Transformer
from predictors.neural_predictor.train import train_model_with_target_indices
from predictors.neural_predictor.predict import predict_n
from utils_projection import projector
from utils_encoder import eca_encoder
from utils_plotting import plot_trajectories, plot_spacetime_diagrams
from utils_data_loader import generate_dataset_with_index_mapping


rule_number = 30
lattice_width = 50
context_length = 2
forecast_length = 50
batch_size = 3
n_epochs = 100
source_data, target_data, new_index_mapping = generate_dataset_with_index_mapping(
    rule_number=rule_number,
    lattice_width=lattice_width,
    batch_size=batch_size,
    context_sequence_length=context_length,
    max_sequence_length=forecast_length
) 
tgt_vocab_size = len(new_index_mapping)

model = Transformer(
    src_vocab_size=lattice_width, 
    tgt_vocab_size=tgt_vocab_size, 
    max_seq_length=forecast_length, 
    src_encoder=eca_encoder,
    tgt_encoder=lambda index,array_size: eca_encoder(
        index=new_index_mapping[index],
        array_size=array_size
    )
)

train_model_with_target_indices( 
   n_epochs=n_epochs,
   model=model,
   x_train=source_data,
   y_train=target_data,
)

predicted_data = predict_n(
    model=model, 
    source=source_data,
    target=target_data[:,:1],
    batch_size=batch_size,
    forecast_horizon=forecast_length-1,
)

predicted_data_encoded = array([
    [
        model.encoder_embedding.index_encoder(
            index=new_index_mapping[i],
            array_size=model.encoder_embedding.vocab_size
        ) for i in predicted_data[b]
    ]
    for b in range(batch_size)
])

target_data_encoded=[
    [
        eca_encoder(
            index=new_index_mapping[i],
            array_size=lattice_width
        ) for i in target_data[b]
    ]
    for b in range(batch_size)
]

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



test_source_data, test_target_data, test_new_index_mapping = generate_dataset(
    rule_number=rule_number,
    lattice_width=lattice_width,
    batch_size=batch_size,
    context_sequence_length=context_length,
    max_sequence_length=forecast_length
) 

test_predicted_data = predict_n(
    model=model, 
    source=test_source_data,
    target=test_target_data[:,:1],
    batch_size=batch_size,
    forecast_horizon=forecast_length-1,
)

test_predicted_data_encoded = array([
    [
        model.encoder_embedding.index_encoder(
            index=new_index_mapping[i],
            array_size=model.encoder_embedding.vocab_size
        ) for i in test_predicted_data[b]
    ]
    for b in range(batch_size)
])

test_target_data_encoded=[
    [
        eca_encoder(
            index=test_new_index_mapping[i],
            array_size=lattice_width
        ) for i in test_target_data[b]
    ]
    for b in range(batch_size)
]

plot_trajectories(
    target=[
        [
            projector(
                embedding=embedding,
                lattice_width=lattice_width
            ) for embedding in test_target_data_encoded[b]
        ]
        for b in range(batch_size)
    ], 
    predicted=[
        [
            projector(
                embedding=embedding,
                lattice_width=lattice_width
            ) for embedding in test_predicted_data_encoded[b]
        ]
        for b in range(batch_size)
    ],
    batch_size=batch_size
)

plot_spacetime_diagrams(
    target=test_target_data_encoded,
    predicted=test_predicted_data_encoded,
    batch_size=batch_size
)

