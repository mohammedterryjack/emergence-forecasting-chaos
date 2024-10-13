from torch import randint
from numpy import ndarray 

from transformer import Transformer
from train import train_model_with_target_indices, train_model_with_target_embeddings


src_vocab_size = 5000
tgt_vocab_size = 5000
max_seq_length = 100
batch_size = 2 #64
n_epochs = 10

def encoder(index:int, array_size:int) -> ndarray:
    #TODO: change this to the custom encoding for the dynamical system
    from numpy import zeros

    onehot = zeros(array_size)
    if index < array_size:
        onehot[index] = 1
    return onehot

def decoder(vector:ndarray) -> int:
    """Decode an array of floats (predictions)
    to an integer value 
    by binarising the array using a threshold
    and converting the binary array to its decimal equivalent"""
    threshold = 0.5
    binary_array = vector > threshold
    binary_str = ''.join(binary_array.astype(str))
    return int(binary_str, 2)

from torch import Tensor, tensor, zeros
def encode_data(data:Tensor, vocab_size:int) -> Tensor:
    batch_size, seq_length = data.size()
    data_encoded = zeros((batch_size, seq_length, vocab_size))
    for i in range(batch_size):
        for j in range(seq_length):
            data_encoded[i][j] = tensor(
                encoder(index=data[i][j], array_size=vocab_size)
            )
    return data_encoded

model = Transformer(
    src_vocab_size=src_vocab_size, 
    tgt_vocab_size=tgt_vocab_size, 
    max_seq_length=max_seq_length, 
    src_encoder=encoder,
)

source_data = randint(1, src_vocab_size, (batch_size, max_seq_length))  
target_data = randint(1, tgt_vocab_size, (batch_size, max_seq_length))  
target_data_encoded = encode_data(data=target_data, vocab_size=tgt_vocab_size)
print(target_data_encoded.shape)

#a = source_data[0]
#print(a) #>>100
#x = model.encoder_embedding(source_data)
#print(x[0]) #>>100,512
#b = model.decoder_embedding(x)
#print(b)

train_model_with_target_embeddings(
    n_epochs=n_epochs,
    model=model,
    x_train=source_data,
    y_train=target_data,
    y_train_embedded=target_data_encoded,
    y_vocab_size=tgt_vocab_size
)
# train_model_with_target_indices(
#     n_epochs=n_epochs,
#     model=model,
#     x_train=source_data,
#     y_train=target_data,
#     y_vocab_size=tgt_vocab_size
# )
#output = model(source_data, target_data[:, :-1])

#y = target_data[:, 1:].contiguous().view(-1)
#print(y.shape) #>> [198]
#y_hat = output.contiguous().view(-1, tgt_vocab_size)
#print(y_hat.shape) #>>[198, 5000]


#predicts vectors over size of vocab - take argmax to select predicted ids
#ax = output.argmax(-1)
#bx = target_data[:, :-1]