from numpy import ndarray, array, empty, append

from predictors.neural_predictor.transformer import Transformer
from predictors.neural_predictor.train import train_model_with_target_embeddings
from dynamical_system.eca.elementary_cellular_automata import ElementaryCellularAutomata
from utils_projection import projector
from utils_encoder import eca_encoder, eca_decoder
from utils_plotting import plot_spacetime_diagrams_binarised, plot_trajectories

#==========UTILITIES==================

def generate_dataset(
    lattice_width:int,
    batch_size:int,
    max_sequence_length:int
) -> tuple[ndarray, ndarray]:
    before,after = [],[]
    for _ in range(batch_size):
        ca = ElementaryCellularAutomata(
            lattice_width=lattice_width,
            time_steps=max_sequence_length*2,
            transition_rule_number=3
        )
        metadata = ca.info()
        before.append(metadata.lattice_evolution[:max_sequence_length])
        after.append(metadata.lattice_evolution[max_sequence_length:])
    return array(before),array(after)




def predict_n(
    model:Transformer, 
    sequence:ndarray, 
    max_sequence_length:int, 
    batch_size:int,
    forecast_horizon:int,
    lattice_width:int,
    decoder:callable
) -> tuple[ndarray,ndarray]:
    """autoregressively predict next n steps in sequence"""

    delta = 0
    predictions = empty((batch_size, forecast_horizon, 1))
    predictions_embedded = empty((batch_size, forecast_horizon, lattice_width))
    for iteration in range(forecast_horizon):

        _,sequence_length = sequence.shape
        if sequence_length > max_sequence_length:
            delta = sequence_length - max_sequence_length
        context = array([
            sequence[b,delta:] for b in range(batch_size)
        ])

        predictions_embedded[:,iteration,:] = model.predict_next(
            sequence=context, 
            return_distribution=True
        )[:,0,:]

        predicted_next_indexes = array([[
            decoder(predictions_embedded[b,iteration])
        ] for b in range(batch_size)])

        predictions[:,iteration,:] = predicted_next_indexes

        sequence = array([
            append(sequence[b], predicted_next_indexes)
            for b in range(batch_size)
        ])

    return predictions, predictions_embedded


#==========SETUP==================

src_vocab_size = tgt_vocab_size = 50
max_seq_length = 50
batch_size = 1
n_epochs = 100
binary_threshold = 0.5
source_data, target_data = generate_dataset(
    lattice_width=tgt_vocab_size,
    batch_size=batch_size,
    max_sequence_length=max_seq_length
) 

model = Transformer(
    src_vocab_size=src_vocab_size, 
    tgt_vocab_size=tgt_vocab_size, 
    max_seq_length=max_seq_length, 
    src_encoder=eca_encoder,
)


#==========TRAINING==================
train_model_with_target_embeddings(
  n_epochs=n_epochs,
  model=model,
  x_train=source_data,
  y_train=target_data,
)


#==========PREDICTIONS=============
_,predicted_data_encoded = predict_n(
    model=model, 
    sequence=source_data[:,:1], 
    max_sequence_length=max_seq_length, 
    batch_size=batch_size,
    lattice_width=src_vocab_size,
    forecast_horizon=max_seq_length,
    decoder=lambda embedding:eca_decoder(
        lattice=embedding,
        binary_threshold=binary_threshold
    )
)

#======DISPLAY PREDICTIONS================
source_data_encoded=[
    [
        eca_encoder(index=i,array_size=tgt_vocab_size) for i in source_data[b]
    ]
    for b in range(batch_size)
]
target_data_encoded=[
    [
        eca_encoder(index=i,array_size=tgt_vocab_size) for i in target_data[b]
    ]
    for b in range(batch_size)
]
plot_spacetime_diagrams_binarised(
    source=source_data_encoded,
    target=target_data_encoded,
    predicted=predicted_data_encoded,
    batch_size=batch_size,
    binary_threshold=binary_threshold
)


plot_trajectories(
    target=[
        [
            projector(
                embedding=embedding,
                lattice_width=tgt_vocab_size
            ) for embedding in target_data_encoded[b]
        ]
        for b in range(batch_size)
    ], 
    predicted=[
        [
            projector(
                embedding=embedding,
                lattice_width=tgt_vocab_size
            ) for embedding in predicted_data_encoded[b]
        ]
        for b in range(batch_size)
    ],
    batch_size=batch_size
)



#TODO:
# - investigate why prediction is so bad! is the context correct each time in predict_n?
# - try predicting by adding additional / emergent features
