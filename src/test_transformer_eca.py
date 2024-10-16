from numpy import ndarray, array

from predictors.neural_predictor.transformer import Transformer
from predictors.neural_predictor.train import train_model_with_target_embeddings
from dynamical_system.eca.elementary_cellular_automata import ElementaryCellularAutomata

def generate_dataset(
    src_vocab_size:int,
    tgt_vocab_size:int,
    batch_size:int,
    max_sequence_length:int
) -> tuple[ndarray, ndarray]:
    before,after = [],[]
    for _ in range(batch_size):
        ca = ElementaryCellularAutomata(
            lattice_width=17,
            time_steps=max_sequence_length*2,
            transition_rule_number=90
        )
        metadata = ca.info()
        before.append(metadata.lattice_evolution[:max_sequence_length])
        after.append(metadata.lattice_evolution[max_sequence_length:])
    return array(before),array(after)


#def encoder(index:int, array_size:int) -> ndarray:
#    onehot = zeros(array_size)
#    if index < array_size:
#        onehot[index] = 1
#    return onehot



src_vocab_size = 5000
tgt_vocab_size = 5000
max_seq_length = 100
batch_size = 2
n_epochs = 10
source_data, target_data = generate_dataset(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    batch_size=batch_size,
    max_sequence_length=max_seq_length
) 

# model = Transformer(
#     src_vocab_size=src_vocab_size, 
#     tgt_vocab_size=tgt_vocab_size, 
#     max_seq_length=max_seq_length, 
#     src_encoder=encoder,
# )

# train_model_with_target_embeddings(
#     n_epochs=n_epochs,
#     model=model,
#     x_train=source_data,
#     y_train=target_data,
# )

# print(source_data.shape, target_data.shape)
# predicted_next = model.predict_next(sequence=source_data, return_distribution=True)
# print(predicted_next)
