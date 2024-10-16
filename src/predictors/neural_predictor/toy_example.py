from numpy import ndarray, zeros
from numpy.random import randint 

from predictors.neural_predictor.transformer import Transformer
from predictors.neural_predictor.train import train_model_with_target_embeddings


def toy_encoder(index:int, array_size:int) -> ndarray:
    onehot = zeros(array_size)
    if index < array_size:
        onehot[index] = 1
    return onehot

def toy_dataset(
    src_vocab_size:int,
    tgt_vocab_size:int,
    batch_size:int,
    max_sequence_length:int
) -> tuple[ndarray, ndarray]:
    return (
        randint(1, src_vocab_size, (batch_size, max_sequence_length))  ,
        randint(1, tgt_vocab_size, (batch_size, max_sequence_length)) 
    )

src_vocab_size = 5000
tgt_vocab_size = 5000
max_seq_length = 100
batch_size = 2
n_epochs = 10
source_data, target_data = toy_dataset(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    batch_size=batch_size,
    max_sequence_length=max_seq_length
) 

model = Transformer(
    src_vocab_size=src_vocab_size, 
    tgt_vocab_size=tgt_vocab_size, 
    max_seq_length=max_seq_length, 
    src_encoder=toy_encoder,
)

train_model_with_target_embeddings(
    n_epochs=n_epochs,
    model=model,
    x_train=source_data,
    y_train=target_data,
)

print(source_data.shape, target_data.shape)
predicted_next = model.predict_next(sequence=source_data, return_distribution=True)
print(predicted_next)
