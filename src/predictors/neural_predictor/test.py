from torch import randint
from numpy import ndarray 

from transformer import Transformer
from train import train_model


src_vocab_size = 5000
tgt_vocab_size = 5000
max_seq_length = 100
batch_size = 2 #64
n_epochs = 10

def encoder(index:int, array_size:int) -> ndarray:
    from numpy import zeros

    onehot = zeros(array_size)
    if index < array_size:
        onehot[index] = 1
    return onehot

def decoder(vector:ndarray) -> int:
    #TODO
    return -1

model = Transformer(
    src_vocab_size=src_vocab_size, 
    tgt_vocab_size=tgt_vocab_size, 
    max_seq_length=max_seq_length, 
    src_encoder=encoder,
    tgt_decoder=decoder
)

source_data = randint(1, src_vocab_size, (batch_size, max_seq_length))  
target_data = randint(1, tgt_vocab_size, (batch_size, max_seq_length))  

#a = source_data[0]
#print(a) #>>100
#x = model.encoder_embedding(source_data)
#print(x[0]) #>>100,512
#b = model.decoder_embedding(x)
#print(b)


# train_model(
#     n_epochs=n_epochs,
#     model=model,
#     x_train=source_data,
#     y_train=target_data,
#     y_vocab_size=tgt_vocab_size
# )

#output = model(source_data, target_data[:, :-1])
#predicts vectors over size of vocab - take argmax to select predicted ids
#ax = output.argmax(-1)
#bx = target_data[:, :-1]