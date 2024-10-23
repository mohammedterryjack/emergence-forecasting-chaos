from numpy import ndarray, array, empty, append

from predictors.neural_predictor.transformer import Transformer
from predictors.neural_predictor.train import train_model_with_target_embeddings
from dynamical_system.eca.elementary_cellular_automata import ElementaryCellularAutomata


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
            transition_rule_number=1
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
    sequence=source_data, 
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

from matplotlib.pyplot import subplots, show, tight_layout, legend

_, axes = subplots(batch_size, 4, figsize=(10, 5 * batch_size))
for b in range(batch_size):

    axes[b, 0].imshow(source_data_encoded[b], cmap='gray')
    axes[b, 0].set_title(f'Source {b+1}')
    axes[b, 0].axis('off') 

    axes[b, 1].imshow(target_data_encoded[b], cmap='gray')
    axes[b, 1].set_title(f'Target {b+1}')
    axes[b, 1].axis('off') 

    axes[b, 2].imshow(predicted_data_encoded[b] > binary_threshold, cmap='gray')
    axes[b, 2].set_title(f'Predicted (Binarised) {b+1}')
    axes[b, 2].axis('off') 

    axes[b, 3].imshow(predicted_data_encoded[b], cmap='gray')
    axes[b, 3].set_title(f'Predicted (Real) {b+1}')
    axes[b, 3].axis('off') 

tight_layout()
show()


from numpy.linalg import norm
from numpy import arccos, ones

def convert_zeros_to_minus_one(x:ndarray) -> ndarray:
    return 2*x - 1

def cosine_similarity(a:ndarray, b:ndarray) -> float:
    result = a @ b.T
    result /= (norm(a)*norm(b))+1e-9
    return result

def angle(x:ndarray, origin:ndarray) -> float:
    cos_theta = cosine_similarity(a=origin,b=convert_zeros_to_minus_one(x))
    return arccos(cos_theta)

def projector(embedding:ndarray, lattice_width:int) -> float:
    ref_point = ones(shape=(lattice_width))
    return angle(x=embedding,origin=ref_point)

predicted_data_projected = [
    [
        projector(embedding=embedding,lattice_width=tgt_vocab_size) for embedding in predicted_data_encoded[b]
    ]
    for b in range(batch_size)
]
target_data_projected = [
    [
        projector(embedding=embedding,lattice_width=tgt_vocab_size) for embedding in target_data_encoded[b]
    ]
    for b in range(batch_size)
]


_, axes = subplots(batch_size, 1, figsize=(10, 5 * batch_size))
for b in range(batch_size):
    axes[b].set_title(f'Batch {b+1}')
    axes[b].plot(target_data_projected[b], label='expected', color='b')
    axes[b].plot(predicted_data_projected[b], label='predicted', linestyle=':', color='g')

legend()
tight_layout()
show()


#TODO:
# - try predicting by adding additional / emergent features
# - eca test with non-neural model
