from numpy import ndarray, array, empty, append

from predictors.neural_predictor.transformer import Transformer
from predictors.neural_predictor.train import train_model_with_target_embeddings
from dynamical_system.eca.elementary_cellular_automata import ElementaryCellularAutomata

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
            transition_rule_number=90
        )
        metadata = ca.info()
        before.append(metadata.lattice_evolution[:max_sequence_length])
        after.append(metadata.lattice_evolution[max_sequence_length:])
    return array(before),array(after)#, metadata.lattice_configuration_space


def eca_encoder(index:int, array_size:int) -> ndarray:
    return array(ElementaryCellularAutomata.create_binary_lattice_from_number(
        state_number=index,
        lattice_width=array_size
    ))

def eca_decoder(lattice:list[float], binary_threshold:float) -> int:
    binary_lattice = (lattice>binary_threshold).astype(int).tolist()
    return ElementaryCellularAutomata.get_state_number_from_binary_lattice(
        binary_lattice=binary_lattice
    )

def predict_n(
    model:Transformer, 
    sequence:ndarray, 
    max_sequence_length:int, 
    batch_size:int,
    forecast_horizon:int,
    lattice_width:int,
    binary_threshold:float
) -> ndarray:
    """autoregressively predict next n steps in sequence"""

    delta = 0
    predictions = empty((batch_size, forecast_horizon, lattice_width))
    for iteration in range(forecast_horizon):

        _,sequence_length = sequence.shape
        if sequence_length > max_sequence_length:
            delta = sequence_length - max_sequence_length
        context = array([
            sequence[b,delta:] for b in range(batch_size)
        ])

        predictions[:,iteration,:] = model.predict_next(
            sequence=context, 
            return_distribution=True
        )[:,0,:]

        predicted_next_indexes = array([
            [eca_decoder(
                lattice=predictions[b,iteration],
                binary_threshold=binary_threshold
            )] for b in range(batch_size)
        ])

        sequence = array([
            append(sequence[b], predicted_next_indexes)
            for b in range(batch_size)
        ])

    return predictions

src_vocab_size = tgt_vocab_size = 17
max_seq_length = 100
batch_size = 2
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

train_model_with_target_embeddings(
    n_epochs=n_epochs,
    model=model,
    x_train=source_data,
    y_train=target_data,
)


predicted = predict_n(
    model=model, 
    sequence=source_data, 
    max_sequence_length=max_seq_length, 
    batch_size=batch_size,
    lattice_width=src_vocab_size,
    binary_threshold=binary_threshold,
    forecast_horizon=max_seq_length,
)
source_data_encoded = [
    [
        eca_encoder(index=i,array_size=tgt_vocab_size) for i in source_data[b]
    ]
    for b in range(batch_size)
]
target_data_encoded = [
    [
        eca_encoder(index=i,array_size=tgt_vocab_size) for i in target_data[b]
    ]
    for b in range(batch_size)
]

from matplotlib.pyplot import subplots, show, tight_layout
num_rows = batch_size
num_cols = 4

for b in range(batch_size):
    _, axes = subplots(1, num_cols, figsize=(10, 5 * num_rows))

    axes[0].imshow(source_data_encoded[b], cmap='gray')
    axes[0].set_title(f'Source {b+1}')
    axes[0].axis('off') 

    axes[1].imshow(target_data_encoded[b], cmap='gray')
    axes[1].set_title(f'Target {b+1}')
    axes[1].axis('off') 

    axes[2].imshow(predicted[b] > binary_threshold, cmap='gray')
    axes[2].set_title(f'Predicted (Binarised) {b+1}')
    axes[2].axis('off') 

    axes[3].imshow(predicted[b], cmap='gray')
    axes[3].set_title(f'Predicted (Real) {b+1}')
    axes[3].axis('off') 

    tight_layout()
    show()