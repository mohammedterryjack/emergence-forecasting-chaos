from numpy import ndarray, array, append

from predictors.neural_predictor.transformer import Transformer
from predictors.neural_predictor.train import train_model_with_target_indices, train_model_with_target_embeddings
from predictors.neural_predictor.predict import predict_n, predict_n_encoded
from dynamical_system.eca.elementary_cellular_automata import ElementaryCellularAutomata
from utils_projection import projector
from utils_encoder import eca_encoder, eca_decoder
from utils_plotting import plot_trajectories, plot_spacetime_diagrams
#==========UTILITIES==================

def generate_dataset(
    rule_number:int,
    lattice_width:int,
    batch_size:int,
    context_sequence_length:int,
    max_sequence_length:int
) -> tuple[ndarray, ndarray, list[int]]:
    
    cas = [
        ElementaryCellularAutomata(
            lattice_width=lattice_width,
            time_steps=context_sequence_length + max_sequence_length,
            transition_rule_number=rule_number
        ) for _ in range(batch_size)
    ]

    original_to_mini_index_mapping = set()
    for ca in cas:
        original_to_mini_index_mapping |= set(ca.info().lattice_evolution)
    original_to_mini_index_mapping = list(original_to_mini_index_mapping)
    original_to_mini_index_mapping.sort()


    before,after = [],[]
    for ca in cas:
        metadata = ca.info()
        before.append(metadata.lattice_evolution[:context_sequence_length])
        after_ = metadata.lattice_evolution[context_sequence_length:]
        after_encoded = [
            original_to_mini_index_mapping.index(index) for index in after_
        ]
        after.append(after_encoded)
    
    return array(before),array(after),original_to_mini_index_mapping



#==========SETUP==================

rule_number = 3
lattice_width = 50
context_length = 2
forecast_length = 50
batch_size = 3
n_epochs = 100
source_data, target_data, new_index_mapping = generate_dataset(
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

#==========TRAINING==================

train_model_with_target_indices( 
    n_epochs=n_epochs,
    model=model,
    x_train=source_data,
    y_train=target_data,
)

#==========PREDICTIONS=============
predicted_data = predict_n(
    model=model, 
    source=source_data,
    target=target_data[:,:1],
    batch_size=batch_size,
    forecast_horizon=forecast_length-1,
)

# #======DISPLAY PREDICTIONS================
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

#===========NEW MODEL USING DIFFERENT DECODER===============
model2 = Transformer(
    src_vocab_size=lattice_width, 
    tgt_vocab_size=lattice_width, 
    max_seq_length=forecast_length, 
    src_encoder=eca_encoder,
    tgt_encoder=eca_encoder
)

#==========TRAINING==================

train_model_with_target_embeddings( 
   n_epochs=n_epochs,
   model=model2,
   x_train=source_data,
   y_train=target_data,
)

#==========PREDICTIONS=============
predicted_data2 = predict_n_encoded(
    model=model2, 
    source=source_data,
    target=target_data[:,:1],
    batch_size=batch_size,
    forecast_horizon=forecast_length-1,
    vector_to_index=lambda vector: eca_decoder(
        lattice=vector,
        binary_threshold=0.0
    )
)

#======DISPLAY PREDICTIONS================
predicted_data2_encoded = array([
    [
        model.encoder_embedding.index_encoder(
            index=i,
            array_size=model.encoder_embedding.vocab_size
        ) for i in predicted_data2[b]
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
            ) for embedding in predicted_data2_encoded[b]
        ]
        for b in range(batch_size)
    ],
    batch_size=batch_size
)

plot_spacetime_diagrams(
    target=target_data_encoded,
    predicted=predicted_data2_encoded,
    batch_size=batch_size
)

#TODO:
# - try predicting by adding additional / emergent features in src_encoder
