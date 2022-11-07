import tensorflow as tf
from settings_train import EMBED_DIM, IMAGE_SIZE, SEQ_LENGTH, NUM_LAYERS
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.applications import efficientnet, resnet
import numpy as np


# Get CNN model
def get_cnn_model(selected_cnn_model):
    if selected_cnn_model == "efficientnet":
        base_model = efficientnet.EfficientNetB0(
            include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3)
        )
        # freeze feature extractor layers
        base_model.trainable = False
        base_model_out = base_model.output
        base_model_out = layers.Reshape((-1, 1280))(base_model_out)
        cnn_model = keras.models.Model(base_model.input, base_model_out)
    elif selected_cnn_model == "resnet":
        base_model = resnet.ResNet101(
            include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3)
        )
        # freeze feature extractor layers
        base_model.trainable = False
        base_model_out = base_model.output
        base_model_out = layers.Reshape((-1, 2048))(base_model_out)
        cnn_model = keras.models.Model(base_model.input, base_model_out)
    return cnn_model


# Add & Norm Layer
class AddNormalization(layers.Layer):
    def __init__(self, **kwargs):
        super(AddNormalization, self).__init__(**kwargs)
        self.layer_norm = layers.LayerNormalization()

    def call(self, x):
        return self.layer_norm(x)


# Feed Forward Layer
class FeedForward(layers.Layer):
    def __init__(self, embed_dim, ff_dim, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.dense_1 = layers.Dense(units=ff_dim)
        self.dense_2 = layers.Dense(units=embed_dim)
        self.relu = layers.ReLU()

    def call(self, x):
        return self.dense_2(self.relu(self.dense_1(x)))


# Positional Embedding
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        word_embedding_matrix = self.get_position_encoding(vocab_size, embed_dim)
        position_embedding_matrix = self.get_position_encoding(
            sequence_length, embed_dim
        )
        self.word_embedding_layer = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            weights=[word_embedding_matrix],
            trainable=False,
        )
        self.position_embedding_layer = layers.Embedding(
            input_dim=sequence_length,
            output_dim=embed_dim,
            weights=[position_embedding_matrix],
            trainable=False,
        )

    def get_position_encoding(self, seq_len, d, n=10000):
        P = np.zeros((seq_len, d))
        for pos in range(seq_len):
            for i in np.arange(int(d / 2)):
                denominator = np.power(n, 2 * i / d)
                P[pos, 2 * i] = np.sin(pos / denominator)
                P[pos, 2 * i + 1] = np.cos(pos / denominator)
        return P

    def call(self, x):
        position_indices = tf.range(tf.shape(x)[-1])
        embedded_words = self.word_embedding_layer(x)
        embedded_indices = self.position_embedding_layer(position_indices)
        return embedded_words + embedded_indices


# Encoder Layer
# class EncoderLayer(layers.Layer):
#     def __init__(self, embed_dim, ff_dim, num_heads, dropout_rate=0.1, **kwargs):
#         super(EncoderLayer, self).__init__(**kwargs)
#         self.multihead_attention = layers.MultiHeadAttention(
#             num_heads=num_heads, key_dim=embed_dim
#         )
#         self.add_norm_1 = AddNormalization()
#         self.dropout_1 = layers.Dropout(dropout_rate)
#         self.feed_forward = FeedForward(embed_dim, ff_dim)
#         self.add_norm_2 = AddNormalization()
#         self.dropout_2 = layers.Dropout(dropout_rate)

#     def call(self, inputs, training):
#         multihead_ouput = self.multihead_attention(
#             query=inputs, value=inputs, key=inputs
#         )
#         multihead_ouput = self.dropout_1(multihead_ouput, training=training)
#         out1 = self.add_norm_1(inputs + multihead_ouput)
#         feed_forward_output = self.feed_forward(out1)
#         feed_forward_output = self.dropout_2(feed_forward_output, training=training)
#         return self.add_norm_2(out1 + feed_forward_output)


# Encoder
class Encoder(layers.Layer):
    def __init__(
        self, num_layers, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs
    ):
        super(Encoder, self).__init__(**kwargs)
        self.multihead_attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense = layers.Dense(embed_dim, activation="relu")
        self.add_norm = layers.LayerNormalization()

    def call(self, inputs, training, mask=None):
        inputs = self.dense(inputs)
        multihead_ouput = self.multihead_attention(
            query=inputs,
            value=inputs,
            key=inputs,
            training=training,
            attention_mask=mask,
        )
        enc_output = self.add_norm(inputs + multihead_ouput)
        return enc_output


# Decoder Layer
# class DecoderLayer(layers.Layer):
# def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
#     super(DecoderLayer, self).__init__(**kwargs)
#     self.multihead_attention_1 = layers.MultiHeadAttention(
#         num_heads=num_heads, key_dim=embed_dim
#     )
#     self.multihead_attention_2 = layers.MultiHeadAttention(
#         num_heads=num_heads, key_dim=embed_dim
#     )
#     self.feed_forward = FeedForward(embed_dim, ff_dim)
#     self.add_norm1 = AddNormalization()
#     self.add_norm2 = AddNormalization()
#     self.add_norm3 = AddNormalization()
#     self.dropout_1 = layers.Dropout(dropout_rate)
#     self.dropout_2 = layers.Dropout(dropout_rate)
#     self.dropout_3 = layers.Dropout(dropout_rate)

# def call(self, x, enc_output, training):
#     multihead_output_1 = self.multihead_attention_1(x, x)
#     multihead_output_1 = self.dropout_1(multihead_output_1, training=training)
#     out1 = self.add_norm1(multihead_output_1 + x)
#     multihead_output_2 = self.multihead_attention_2(
#         out1, enc_output, value=enc_output
#     )
#     multihead_output_2 = self.dropout_2(multihead_output_2, training=training)
#     out2 = self.add_norm2(multihead_output_2 + out1)
#     ffn_output = self.feed_forward(out2)
#     ffn_output = self.dropout_3(ffn_output, training=training)
#     return self.add_norm3(ffn_output + out2)


# Decoder
class Decoder(layers.Layer):
    def __init__(
        self,
        num_layers,
        embed_dim,
        num_heads,
        ff_dim,
        vocab_size,
        dropout_rate=0.1,
        **kwargs
    ):
        super(Decoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.pos_encoding = PositionalEmbedding(
            sequence_length=SEQ_LENGTH, vocab_size=vocab_size, embed_dim=EMBED_DIM
        )
        self.multihead_attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.multihead_attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.feed_forward = FeedForward(embed_dim, ff_dim)
        self.add_norm1 = AddNormalization()
        self.add_norm2 = AddNormalization()
        self.add_norm3 = AddNormalization()
        self.dropout_1 = layers.Dropout(dropout_rate)
        self.dropout_2 = layers.Dropout(0.5)
        self.output_layer = layers.Dense(vocab_size)

    def call(self, x, enc_output, training):
        x = self.pos_encoding(x)
        x = self.dropout_1(x, training=training)

        multihead_output1 = self.multihead_attention_1(
            query=x, value=x, key=x, training=training
        )
        out1 = self.add_norm1(multihead_output1 + x)

        multihead_output2 = self.multihead_attention_2(
            query=out1,
            value=enc_output,
            key=enc_output,
            training=training,
        )
        out2 = self.add_norm2(multihead_output2 + out1)

        ffn_output = self.feed_forward(out2)
        out3 = self.add_norm3(ffn_output + out2)

        out3 = self.dropout_2(out3, training=training)
        dec_output = self.output_layer(out3)
        return dec_output


# Image captioning model
class ImageCaptioningModel(keras.Model):
    def __init__(self, cnn_model, encoder, decoder, num_captions_per_image=5, **kwargs):
        super(ImageCaptioningModel, self).__init__(**kwargs)
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="accuracy")
        self.num_captions_per_image = num_captions_per_image

    def call(self, inputs):
        features = self.cnn_model(inputs[0])
        enc_output = self.encoder(features, False)
        dec_output = self.decoder(inputs[2], enc_output, training=inputs[1])
        return dec_output

    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def calculate_accuracy(self, y_true, y_pred, mask):
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

    def train_step(self, batch_data):
        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0

        # 1. Get image embeddings
        img_embed = self.cnn_model(batch_img)

        # 2. Pass each of the five captions one by one to the decoder
        # along with the encoder outputs and compute the loss as well as accuracy
        # for each caption.
        for i in range(self.num_captions_per_image):
            with tf.GradientTape() as tape:
                # 3. Pass image embeddings to encoder
                encoder_out = self.encoder(img_embed, training=True)

                batch_seq_inp = batch_seq[:, i, :-1]
                batch_seq_true = batch_seq[:, i, 1:]

                # 4. Compute the mask for the input sequence
                mask = tf.math.not_equal(batch_seq_inp, 0)

                # 5. Pass the encoder outputs, sequence inputs along with
                # mask to the decoder
                batch_seq_pred = self.decoder(
                    batch_seq_inp, encoder_out, training=True, mask=mask
                )

                # 6. Calculate loss and accuracy
                caption_loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
                caption_acc = self.calculate_accuracy(
                    batch_seq_true, batch_seq_pred, mask
                )

                # 7. Update the batch loss and batch accuracy
                batch_loss += caption_loss
                batch_acc += caption_acc

            # 8. Get the list of all the trainable weights
            train_vars = (
                self.encoder.trainable_variables + self.decoder.trainable_variables
            )

            # 9. Get the gradients
            grads = tape.gradient(caption_loss, train_vars)

            # 10. Update the trainable weights
            self.optimizer.apply_gradients(zip(grads, train_vars))

        loss = batch_loss
        acc = batch_acc / float(self.num_captions_per_image)

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    def test_step(self, batch_data):
        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0

        # 1. Get image embeddings
        img_embed = self.cnn_model(batch_img)

        # 2. Pass each of the five captions one by one to the decoder
        # along with the encoder outputs and compute the loss as well as accuracy
        # for each caption.
        for i in range(self.num_captions_per_image):
            # 3. Pass image embeddings to encoder
            encoder_out = self.encoder(img_embed, training=False)

            batch_seq_inp = batch_seq[:, i, :-1]
            batch_seq_true = batch_seq[:, i, 1:]

            # 4. Compute the mask for the input sequence
            mask = tf.math.not_equal(batch_seq_inp, 0)

            # 5. Pass the encoder outputs, sequence inputs along with
            # mask to the decoder
            batch_seq_pred = self.decoder(
                batch_seq_inp, encoder_out, training=False, mask=mask
            )

            # 6. Calculate loss and accuracy
            caption_loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
            caption_acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)

            # 7. Update the batch loss and batch accuracy
            batch_loss += caption_loss
            batch_acc += caption_acc

        loss = batch_loss
        acc = batch_acc / float(self.num_captions_per_image)

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    @property
    def metrics(self):
        # we need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker, self.acc_tracker]
