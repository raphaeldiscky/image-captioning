import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorflow as tf
from tensorflow import convert_to_tensor
from tensorflow.data import Dataset
from tensorflow.keras.layers import TextVectorization

from datasets import custom_standardization
from positional_embedding import PositionalEmbedding

# input: text_vectorization
# output: sum of both word embedding and the position embedding (attention_is_all_you_need)

output_sequence_length = 25
vocab_size = 20000
sentences = [
    "<start> Sebuah restoran memiliki meja dan kursi kayu modern. <end>",
    "<start> Sebuah meja restoran panjang dengan kursi bulat rotan. <end>",
]

# TEXT VECTORIZATION LAYER
sentence_data = Dataset.from_tensor_slices(sentences)
# create the TextVectorization layer
vectorize_layer = TextVectorization(
    output_mode="int",
    output_sequence_length=output_sequence_length,
    max_tokens=vocab_size,
    standardize=custom_standardization,
)

# train the layer to create a dictionary of words and replaces each word it its corresponding index in the dictionary
vectorize_layer.adapt(sentence_data)
# convert all sentences to tensors
word_tensors = convert_to_tensor(sentences, dtype=tf.string)
# use the word tensors to get vectorized phrases
vectorized_words = vectorize_layer(word_tensors)

positional_embedding = PositionalEmbedding(
    output_sequence_length, vocab_size, output_sequence_length
)

print("\n\nVECTORIZED WORDS", vectorized_words)

attnisallyouneed_output = positional_embedding(vectorized_words)
print("\n\nOutput from my_embedded_layer: ", attnisallyouneed_output)
