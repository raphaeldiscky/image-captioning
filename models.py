import tensorflow as tf
from settings_train import EMBED_DIM, IMAGE_SIZE, SEQ_LENGTH
from tensorflow.keras import layers, Model
from tensorflow import keras
from tensorflow.keras.applications import efficientnet, resnet


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


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)


class Encoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.multihead_attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.addnorm = layers.LayerNormalization()

    def build_graph(self):
        input_layer = layers.Input(shape=(SEQ_LENGTH, self.embed_dim))
        return Model(inputs=[input_layer], outputs=self.call(input_layer))

    def call(self, inputs):
        inputs = self.dense_proj(inputs)
        mha_output = self.multihead_attention(
            query=inputs, value=inputs, key=inputs, attention_mask=None
        )
        enc_output = self.addnorm(inputs + mha_output)
        return enc_output


class Decoder(layers.Layer):
    def __init__(self, embed_dim, ff_dim, num_heads, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.build(input_shape=[None, SEQ_LENGTH, embed_dim])

        self.multihead_attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.multihead_attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.feed_forward = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.addnorm_1 = layers.LayerNormalization()
        self.addnorm_2 = layers.LayerNormalization()
        self.addnorm_3 = layers.LayerNormalization()

        self.embedding = PositionalEmbedding(
            embed_dim=EMBED_DIM, sequence_length=SEQ_LENGTH, vocab_size=self.vocab_size
        )
        self.model_last_layer = layers.Dense(self.vocab_size)
        self.dropout_1 = layers.Dropout(0.1)
        self.dropout_2 = layers.Dropout(0.5)
        self.supports_masking = True

    def build_graph(self):
        input_layer = layers.Input(shape=(SEQ_LENGTH, self.embed_dim))
        return Model(
            inputs=[input_layer],
            outputs=self.call(input_layer, input_layer, True, mask=None),
        )

    def call(self, inputs, encoder_outputs, training, mask=None):
        inputs = self.embedding(inputs)
        causal_mask = self.get_causal_attention_mask(inputs)
        inputs = self.dropout_1(inputs, training=training)

        if mask is not None:
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)
        else:
            combined_mask = None
            padding_mask = None

        mha_output_1 = self.multihead_attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=combined_mask
        )
        addnorm_output1 = self.addnorm_1(inputs + mha_output_1)

        mha_output_2 = self.multihead_attention_2(
            query=addnorm_output1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        addnorm_output2 = self.addnorm_2(addnorm_output1 + mha_output_2)

        ff_output = self.feed_forward(addnorm_output2)
        addnorm_output3 = self.addnorm_3(addnorm_output2 + ff_output)
        addnorm_output3 = self.dropout_2(addnorm_output3, training=training)

        enc_output = self.model_last_layer(addnorm_output3)
        return enc_output

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
        encoder,
        decoder,
        num_captions_per_image=5,
    ):
        super().__init__()
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="accuracy")
        self.num_captions_per_image = num_captions_per_image

    def call(self, inputs):
        x = self.cnn_model(inputs[0])
        x = self.encoder(x, False)
        x = self.decoder(inputs[2], x, training=inputs[1], mask=None)
        return x

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
