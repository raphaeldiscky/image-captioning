import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from numpy import random
from multihead_attention import MultiHeadAttention

input_seq_length = 5  # Maximum length of the input sequence
num_heads = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
embed_dim = 512  # Dimensionality of the model sub-layers' outputs
batch_size = 64  # Batch size from the training process

queries = random.random((batch_size, input_seq_length, d_k))
keys = random.random((batch_size, input_seq_length, d_k))
values = random.random((batch_size, input_seq_length, d_v))

multihead_attention = MultiHeadAttention(num_heads, d_k, d_v, embed_dim)
print("\n\nFinal output: ", multihead_attention(queries, keys, values))
