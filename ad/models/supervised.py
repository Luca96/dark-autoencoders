
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import ad

from tensorflow.keras import layers as tfkl


def get_classifier(input_shape: tuple, conv_filters: list, num_classes: int,
                   activation=tf.nn.leaky_relu, kernel=3, dropout=0.0, include_se=False,
                   **kwargs) -> tf.keras.Model:
    """Defines the default architecture of the encoder"""
    def activation_fn(layer: tfkl.Layer, block: str):
        h = tfkl.BatchNormalization(name=f'batch_norm-b{block}')(layer)
        return tfkl.Activation(activation, name=f'activation-b{block}')(h)

    def conv_block(layer: Layer, filters: int, block: int):
        h = tfkl.SeparableConv2D(filters, kernel_size=kernel, padding='same',
                                 name=f'sep_conv1-b{block}', activation=activation,
                                 **kwargs)(layer)

        if dropout > 0.0:
            h = tfkl.SpatialDropout2D(rate=dropout, name=f'drop-b{block}')(h)

        h = tfkl.SeparableConv2D(filters, kernel_size=kernel, padding='same',
                                 name=f'sep_conv2-b{block}', **kwargs)(h)
        h = activation_fn(h, f'{block}_0')

        if include_se:
            h = ad.layers.SqueezeAndExcite(activation=activation, name=f'SE-b{block}')(h)

        # overlapping Max-pooling
        return tfkl.MaxPool2D(pool_size=3, strides=2, padding='same',
                              name=f'max_pool-b{block}')(h)

    image = tfkl.Input(shape=input_shape, name='image')

    # stem part
    x = tfkl.Conv2D(filters=conv_filters[0], kernel_size=kernel,
                    padding='same', name='conv-stem', **kwargs)(image)
    x = activation_fn(layer=x, block='stem')
    x = tfkl.MaxPool2D(name='max_pool-stem')(x)

    for i, num_filters in enumerate(conv_filters[1:]):
        x = conv_block(layer=x, filters=int(num_filters), block=i)

    # output
    x = tfkl.GlobalAveragePooling2D(name='global_pool')(x)
    y = tfkl.Dense(units=int(num_classes), name='logits')(x)

    return tf.keras.Model(inputs=image, outputs=y, name='Classifier')


# -------------------------------------------------------------------------------------------------
# -- Compact Convolutional Transformer (CCT)
# -- based on: https://keras.io/examples/vision/cct/
# -------------------------------------------------------------------------------------------------

class CCTTokenizer(tfkl.Layer):
    def __init__(self, kernel=3, stride=1, padding=1, pool_kernel=3,
                 pool_stride=2, out_channels=(64, 128),
                 positional_emb=True, **kwargs):
        super().__init__(**kwargs)

        # This is our tokenizer.
        self.tkn_layers = []

        for filters in out_channels:
            conv = tfkl.Conv2D(filters, kernel, stride, padding='valid',
                               use_bias=False, activation=tf.nn.relu,
                               kernel_initializer='he_normal')

            zero_pad = tfkl.ZeroPadding2D(padding)
            max_pool = tfkl.MaxPool2D(pool_kernel, pool_stride, padding='same')

            self.tkn_layers.extend([conv, zero_pad, max_pool])

        self.positional_emb = positional_emb

    def call(self, x: tf.Tensor):
        for layer in self.tkn_layers:
            x = layer(x)

        out = x

        # After passing the images through our mini-network the spatial dimensions
        # are flattened to form sequences.
        reshaped = tf.reshape(
            out,
            (-1, tf.shape(out)[1] * tf.shape(out)[2], tf.shape(out)[-1]))

        return reshaped

    def positional_embedding(self, image_shape: tuple):
        # Here, we calculate the number of sequences and initialize an `Embedding`
        # layer to compute the positional embeddings later.
        if self.positional_emb:
            dummy_inputs = tf.ones((1,) + image_shape)
            dummy_outputs = self.call(dummy_inputs)
            sequence_length = tf.shape(dummy_outputs)[1]
            projection_dim = tf.shape(dummy_outputs)[-1]

            embed_layer = tfkl.Embedding(input_dim=sequence_length,
                                         output_dim=projection_dim)

            return embed_layer, sequence_length

        return None


class CCTDataAugmentation(tf.keras.Model):
    def call(self, x: tf.Tensor, training=False):
        if training:
            return self.augment(x)

        return x

    @tf.function
    def augment(self, x: tf.Tensor):
        return tf.map_fn(fn=ad.aug.tf_augment, elems=x)


def get_cct(num_classes: int, input_shape: tuple, transformer_units: list,
            projection_dim: int, num_heads=2, stochastic_depth_rate=0.1,
            positional=True, dropout=0.1, transformer_layers=2, inspect=False, **kwargs):
    def mlp(x: tfkl.Layer, hidden_units: list, dropout_rate: float):
        for units in hidden_units:
            x = tfkl.Dense(units, activation=tf.nn.gelu)(x)
            x = tfkl.Dropout(dropout_rate)(x)
        return x

    inputs = tfkl.Input(input_shape)
    outputs = []

    # Augment data.
    augmented = CCTDataAugmentation(name='data_aug')(inputs)

    # Encode patches.
    cct_tokenizer = CCTTokenizer(**kwargs)
    encoded_patches = cct_tokenizer(augmented)

    # Apply positional embedding.
    if positional:
        pos_embed, seq_length = cct_tokenizer.positional_embedding(input_shape)
        positions = tf.range(start=0, limit=seq_length, delta=1)
        position_embeddings = pos_embed(positions)
        encoded_patches += position_embeddings

    # Calculate Stochastic Depth probabilities.
    dpr = [x for x in np.linspace(0, stochastic_depth_rate, transformer_layers)]

    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):
        # Layer normalization 1.
        x1 = tfkl.LayerNormalization(epsilon=1e-5)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output, w = tfkl.MultiHeadAttention(num_heads=num_heads,
                                                      key_dim=projection_dim,
                                                      dropout=0.1)(x1, x1, return_attention_scores=True)
        if inspect:
            outputs.append(w)

        # Skip connection 1.
        x2 = tfa.layers.StochasticDepth(dpr[i])([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = tfkl.LayerNormalization(epsilon=1e-5)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=float(dropout))

        # Skip connection 2.
        encoded_patches = tfa.layers.StochasticDepth(dpr[i])([x3, x2])

    # Apply sequence pooling.
    representation = tfkl.LayerNormalization(epsilon=1e-5)(encoded_patches)
    attention_weights = tf.nn.softmax(tfkl.Dense(1)(representation), axis=1)
    weighted_representation = tf.matmul(
        attention_weights, representation, transpose_a=True)
    weighted_representation = tf.squeeze(weighted_representation, -2)

    # Classify outputs.
    logits = tfkl.Dense(num_classes, name='logits')(weighted_representation)

    if not inspect:
        return tf.keras.Model(inputs=inputs, outputs=logits, name='Compact-ViT')
    else:
        return tf.keras.Model(inputs=inputs, outputs=(logits, outputs), name='Compact-ViT')
