from torch import Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import Adam

from transformer import Transformer

def train_model_with_target_indices(n_epochs:int, model:Transformer, x_train:Tensor, y_train:Tensor, y_vocab_size:int) -> None:
    criterion = CrossEntropyLoss(ignore_index=0)
    optimiser = Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    model.train()

    for epoch in range(n_epochs):
        optimiser.zero_grad()
        output = model(x_train, y_train[:, :-1])
        y_hat = output.contiguous().view(-1, y_vocab_size)
        y = y_train[:, 1:].contiguous().view(-1)
        loss = criterion(y_hat, y)
        loss.backward()
        optimiser.step()
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

def train_model_with_target_embeddings(n_epochs:int, model:Transformer, x_train:Tensor, y_train:Tensor, y_train_embedded:Tensor, y_vocab_size:int) -> None:
    criterion = BCEWithLogitsLoss() 
    optimiser = Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    model.train()

    for epoch in range(n_epochs):
        optimiser.zero_grad()
        output = model(x_train, y_train[:, :-1])
        y_hat = output.contiguous().view(-1, y_vocab_size)
        y = y_train_embedded[:, 1:].contiguous().view(-1, y_vocab_size)
        loss = criterion(y_hat, y)
        loss.backward()
        optimiser.step()
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")