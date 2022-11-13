import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorflow as tf
from tensorflow import convert_to_tensor
from tensorflow.keras.layers import TextVectorization, Embedding
from tensorflow.data import Dataset
from datasets import custom_standardization


# input: list of sentences
output_sequence_length = 6
vocab_size = 20
sentences = ["<start> saya adalah robot. <end>", "<start> kamu juga robot. <end>"]

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

print("\n\nVocabulary: ", vectorize_layer.get_vocabulary())
print("\n\nVectorized words: ", vectorized_words)

# EMBEDDING LAYER: convert integers to dense vectors.
# WORD EMBEDDING: output is different because the weights is initially randomized
word_embedding_layer = Embedding(vocab_size, output_sequence_length)
embedded_words = word_embedding_layer(vectorized_words)
print("\n\nEmbedded words: ", embedded_words)

# POSITION EMBEDDING: maximum position is correspond to output_sequence_length
position_embedding_layer = Embedding(output_sequence_length, output_sequence_length)
position_indices = tf.range(output_sequence_length)
embedded_indices = position_embedding_layer(position_indices)
print("\n\nEmbedded indices: ", embedded_indices)

# POSITION ENCODING LAYER: word embedding + position embedding
final_output_embedding = embedded_words + embedded_indices
print("\n\nFinal output: ", final_output_embedding)
