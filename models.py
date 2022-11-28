import tensorflow as tf
from settings_train import EMBED_DIM, IMAGE_SIZE, SEQ_LENGTH
from tensorflow.keras.layers import (
    Layer,
    Reshape,
    LayerNormalization,
    Dense,
    ReLU,
    Dropout,
    MultiHeadAttention,
)
from tensorflow import keras
from tensorflow.keras.applications import efficientnet, resnet
from positional_embedding import PositionalEmbedding


def get_cnn_model(selected_cnn_model):
    if selected_cnn_model == "efficientnet":
        base_model = efficientnet.EfficientNetB0(
            include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3)
        )
        # freeze the convolutional base
        base_model.trainable = False
        base_model_out = base_model.output
        base_model_out = Reshape((-1, 1280))(base_model_out)
        cnn_model = keras.models.Model(base_model.input, base_model_out)
    elif selected_cnn_model == "resnet":
        base_model = resnet.ResNet101(
            include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3)
        )
        # freeze the convolutional base
        base_model.trainable = False
        base_model_out = base_model.output
        base_model_out = Reshape((-1, 2048))(base_model_out)
        cnn_model = keras.models.Model(base_model.input, base_model_out)
    return cnn_model


class AddNormalization(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm = LayerNormalization()

    def call(self, x, sublayer_x):
        add = x + sublayer_x
        return self.layer_norm(add)


class FeedForward(Layer):
    def __init__(self, embed_dim, ff_dim, **kwargs):
        super().__init__(**kwargs)
        self.dense_1 = Dense(ff_dim)
        self.dense_2 = Dense(embed_dim)
        self.relu = ReLU()

    def call(self, inputs):
        x = self.dense_1(inputs)
        return self.dense_2(self.relu(x))


class Encoder(Layer):
    def __init__(self, embed_dim, ff_dim, num_heads, key_dim, value_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.key_dim = key_dim  # key_dim = dimension of key for each head
        self.value_dim = value_dim  # value_dim = embed_dim / num_head
        self.dense = Dense(embed_dim, activation="relu")
        self.multihead_attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=value_dim,
            output_shape=embed_dim,
        )
        self.dropout_1 = Dropout(0.1)
        self.add_norm1 = AddNormalization()
        self.feed_forward = FeedForward(embed_dim, ff_dim)
        self.dropout_2 = Dropout(0.1)
        self.add_norm2 = AddNormalization()

    def call(self, inputs, training):
        # print("\n\n INPUTS ENC:", inputs)  # (None, 64, 1280)
        inputs = self.dense(inputs)
        # print("\n\n DENSE:", inputs)  # (None, 64, 2048)
        multihead_attention_output = self.multihead_attention(
            query=inputs, value=inputs, key=inputs
        )
        # print("\n\n MHA:", multihead_attention_output)  # (None, 64, 2048)
        multihead_attention_output = self.dropout_1(
            multihead_attention_output, training
        )
        # print("\n\n DROP:", multihead_attention_output)
        addnorm_output = self.add_norm1(inputs, multihead_attention_output)
        # print("\n\n NORM:", addnorm_output)  # (None, 64, 2048)
        feed_forward_output = self.feed_forward(addnorm_output)
        # print("\n\n FF:", feed_forward_output)  # (None, 64, 2048)
        feed_forward_output = self.dropout_2(feed_forward_output, training)
        enc_output = self.add_norm2(addnorm_output, feed_forward_output)
        # print("\n\n ENC OUTPUT:", enc_output)  # (None, 64, 2048)
        return enc_output


class Decoder(Layer):
    def __init__(
        self,
        embed_dim,
        ff_dim,
        num_heads,
        vocab_size,
        key_dim,
        value_dim,
        seq_length,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.seq_length = seq_length
        self.multihead_attention_1 = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=value_dim,
            output_shape=embed_dim,
        )
        self.multihead_attention_2 = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=value_dim,
            output_shape=embed_dim,
        )
        self.feed_forward = FeedForward(embed_dim, ff_dim)
        self.add_norm1 = AddNormalization()
        self.add_norm2 = AddNormalization()
        self.add_norm3 = AddNormalization()
        self.pos_encoding = PositionalEmbedding(
            embed_dim=self.embed_dim,
            sequence_length=self.seq_length,
            vocab_size=self.vocab_size,
        )
        self.dropout_1 = Dropout(0.1)
        self.dropout_2 = Dropout(0.1)
        self.dropout_3 = Dropout(0.1)
        self.dense = Dense(self.vocab_size)
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, training, mask=None):
        inputs = self.pos_encoding(inputs)
        # print("\n\nINPUTS DEC:", inputs)  # (None, 24, 2048)
        causal_mask = self.get_causal_attention_mask(inputs)
        # print("\n\nCAUSAL MASK:", causal_mask)  # (None, 24, 24)
        if mask is not None:
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)
        else:
            combined_mask = None
            padding_mask = None
        # print("\n\nPADDING MASK:", padding_mask)  # (None, 24, 24)
        # print("\n\nCOMBINED MASK:", combined_mask)  # (None, 24, 24)
        multihead_output_1 = self.multihead_attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=combined_mask
        )
        # print("\n\n MHA1:", multihead_output_1)  # (None, 24, 2048)
        multihead_output_1 = self.dropout_1(multihead_output_1, training=training)
        addnorm_output_1 = self.add_norm1(inputs, multihead_output_1)
        # print("\n\n ADD NORM 1:", multihead_output_1)  # (None, 24, 2048)
        multihead_output_2 = self.multihead_attention_2(
            query=addnorm_output_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        # print("\n\n MHA2:", multihead_output_1)  # (None, 24, 2048)
        multihead_output_2 = self.dropout_2(multihead_output_2, training=training)
        addnorm_output_2 = self.add_norm2(addnorm_output_1, multihead_output_2)
        # print("\n\n ADD NORM 2:", multihead_output_1)  # (None, 24, 2048)
        ff_output = self.feed_forward(addnorm_output_2)
        # print("\n\n FF:", ff_output)  # (None, 24, 2048)
        ff_output = self.dropout_3(ff_output, training=training)
        addnorm_output_3 = self.add_norm3(addnorm_output_2, ff_output)
        # print("\n\n ADD NORM 3:", multihead_output_1)  # (None, 24, 2048)
        dec_output = self.dense(addnorm_output_3)
        # print("\n\n DEC_OUTPUT:", dec_output)  # (None, 24, 20000)
        return dec_output

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)


class ImageCaptioningModel(keras.Model):
    def __init__(
        self,
        cnn_model,
        embed_dim,
        ff_dim,
        num_heads,
        key_dim,
        value_dim,
        seq_length,
        vocab_size,
    ):
        super().__init__()
        self.cnn_model = get_cnn_model(cnn_model)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.encoder = Encoder(
            embed_dim,
            ff_dim,
            num_heads,
            key_dim,
            value_dim,
        )
        self.decoder = Decoder(
            embed_dim, ff_dim, num_heads, vocab_size, key_dim, value_dim, seq_length
        )

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="accuracy")
        self.num_captions_per_image = 5

    def call(self, inputs):
        enc_input = self.cnn_model(inputs[0])
        # print('\n\nENC_INPUT', enc_input) # (None, 64, 2048)
        enc_output = self.encoder(enc_input, False)
        # print('\n\nENC_INPUT', enc_input) # (None, 64, 2048)
        dec_output = self.decoder(inputs[2], enc_output, training=inputs[1], mask=None)
        # print('\n\nENC_INPUT', enc_input) # (None, 64, 2048)
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
        return [self.loss_tracker, self.acc_tracker]
