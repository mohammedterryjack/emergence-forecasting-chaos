#https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb

from math import sqrt, log

from numpy import ndarray
from torch import matmul, softmax, zeros, arange, exp, float, sin, cos, triu, ones, Tensor, tensor
from torch.nn import Linear, Module, ReLU, LayerNorm, Dropout, Embedding, ModuleList

class MultiHeadAttention(Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = matmul(Q, K.transpose(-2, -1)) / sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = softmax(attn_scores, dim=-1)
        output = matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

class PositionWiseFeedForward(Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)
        self.relu = ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = zeros(max_seq_length, d_model)
        position = arange(0, max_seq_length, dtype=float).unsqueeze(1)
        div_term = exp(arange(0, d_model, 2).float() * -(log(10000.0) / d_model))
        
        pe[:, 0::2] = sin(position * div_term)
        pe[:, 1::2] = cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
class EncoderLayer(Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
class DecoderLayer(Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class InputLayer(Module):
    def __init__(self, vocab_size:int, d_model:int, encoder:callable) -> None:
        super(InputLayer, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.projection_layer = Linear(self.vocab_size, self.d_model, bias=False)
        self.index_encoder = encoder

    def forward(self, input:Tensor) -> Tensor:
        return self.projection_layer(self._encode(input=input))

    def _encode(self, input:Tensor) -> Tensor:
        batch_size, seq_length = input.size()
        input_encoded = zeros((batch_size, seq_length, self.vocab_size))
        for i in range(batch_size):
            for j in range(seq_length):
                index = input[i][j]
                input_encoded[i][j] = tensor(
                    self.index_encoder(index=index, array_size=self.vocab_size)
                )
        return input_encoded


class Transformer(Module):
    def __init__(
        self, 
        src_encoder:callable,
        src_vocab_size:int, 
        tgt_vocab_size:int, 
        max_seq_length:int, 
        d_model:int=512, 
        num_layers:int=6, 
        d_ff:int=2048, 
        num_heads:int=8, 
        dropout:int=0.1
    ) -> None:
        super(Transformer, self).__init__()
        #self.encoder_embedding = Embedding(src_vocab_size, d_model)
        #self.decoder_embedding = Embedding(tgt_vocab_size, d_model)
        self.encoder_embedding = InputLayer(vocab_size=src_vocab_size, d_model=d_model, encoder=src_encoder)
        self.decoder_embedding = InputLayer(vocab_size=tgt_vocab_size, d_model=d_model, encoder=src_encoder)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = Linear(d_model, tgt_vocab_size)
        self.dropout = Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - triu(ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt) 
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output
    
    def predict_next(self, sequence:ndarray, return_distribution:bool=False) -> ndarray:
        """
        Given a sequence of integers as a numpy array
        the model will predict the next in the sequece
        If return_distribution is False
        the output is given as an integer like the input
        otherwise the raw distribution over the target vocab is returned"""
        seq = tensor(sequence)
        predictions = self(seq[:,:-1],seq[:,-1:]).detach().numpy()
        return predictions if return_distribution else predictions.argmax(-1)