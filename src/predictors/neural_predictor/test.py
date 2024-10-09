from torch import randint

from transformer import Transformer
from train import train_model


src_vocab_size = 5000
tgt_vocab_size = 5000
max_seq_length = 100
batch_size = 1 #64
n_epochs = 10

model = Transformer(
    src_vocab_size=src_vocab_size, 
    tgt_vocab_size=tgt_vocab_size, 
    max_seq_length=max_seq_length, 
)

source_data = randint(1, src_vocab_size, (batch_size, max_seq_length))  
target_data = randint(1, tgt_vocab_size, (batch_size, max_seq_length))  

# train_model(
#     n_epochs=n_epochs,
#     model=model,
#     x_train=source_data,
#     y_train=target_data,
#     y_vocab_size=tgt_vocab_size
# )

output = model(source_data, target_data[:, :-1])
#predicts vectors over size of vocab - take argmax to select predicted ids
ax = output.argmax(-1)
bx = target_data[:, :-1]