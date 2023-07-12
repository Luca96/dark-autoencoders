
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp

from tensorflow.keras import layers as tfkl
from typing import List


class Depth2Space(tf.keras.layers.Lambda):
    def __init__(self, ratio=2, name=None, **kwargs):
        assert ratio >= 2
        super().__init__(lambda t: tf.nn.depth_to_space(t, block_size=int(ratio)),
                         name=name, **kwargs)


class Space2Depth(tf.keras.layers.Lambda):
    def __init__(self, ratio=2, name=None, **kwargs):
        assert ratio >= 2
        super().__init__(lambda t: tf.nn.space_to_depth(t, block_size=int(ratio)),
                         name=name, **kwargs)


class L2Normalization(tf.keras.layers.Layer):
    """Normalizes the given tensor to unit norm by dividing by its l2-norm"""

    @tf.function
    def call(self, x: tf.Tensor, **kwargs):
        return x / tf.norm(x, axis=-1, keepdims=True)


class Sampling(tf.keras.layers.Layer):
    """Sampling layer for VAE that implements the re-parametrization trick for differentiable Gaussian sampling"""

    @tf.function
    def call(self, inputs, training=False):
        mean, log_var = inputs

        if training:
            # sample from a Standard Normal
            epsilon = tf.random.normal(shape=tf.shape(mean))

            # Re-parametrization trick
            return mean + tf.exp(0.5 * log_var) * epsilon

        return mean


class SqueezeAndExcite(tf.keras.layers.Layer):
    """Based on https://github.com/titu1994/keras-squeeze-excite-network"""

    def __init__(self, ratio=16, activation='relu', name=None, **kwargs):
        super().__init__(name=name)

        self.ratio = int(ratio)
        self.kwargs = kwargs
        self.activation = activation

        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.reshape = None
        self.dense1 = None
        self.dense2 = None
        self.multiply = tf.keras.layers.Multiply()

    def build(self, input_shape: tuple):
        num_filters = input_shape[-1]

        self.reshape = tf.keras.layers.Reshape(target_shape=(1, 1, num_filters))
        self.dense1 = tf.keras.layers.Dense(units=num_filters // self.ratio, use_bias=False,
                                            activation=self.activation, **self.kwargs)
        self.dense2 = tf.keras.layers.Dense(units=num_filters, activation='sigmoid',
                                            use_bias=False, **self.kwargs)

    @tf.function
    def call(self, x: tf.Tensor):
        h = self.global_pool(x)
        h = self.reshape(h)

        h = self.dense1(h)
        h = self.dense2(h)

        return self.multiply([x, h])


class SpatialBroadcast(tf.keras.layers.Layer):
    """A layer that implements the 'spatial broadcast' operation used in VAE decoder networks.
        - Spatial Broadcast Decoder: https://arxiv.org/pdf/1901.07017
    """

    def __init__(self, width: int, height: int, **kwargs):
        w = int(width)
        h = int(height)

        assert w > 1 and h > 1
        super().__init__(**kwargs)

        self.w = w
        self.h = h

        # create coordinates that will later be concatenated to the tiled latents
        self.tile_shape = (1, h, w, 1)
        self.x_mesh, self.y_mesh = self.get_xy_meshgrid(w, h)

    def call(self, latents, **kwargs):
        batch_size = tf.shape(latents)[0]

        # tile the latent vectors
        z = tf.reshape(latents, shape=(batch_size, 1, 1, -1))
        z = tf.tile(z, multiples=self.tile_shape)

        # also tile the xy-meshgrid
        x = tf.tile(self.x_mesh, multiples=(batch_size, 1, 1, 1))
        y = tf.tile(self.y_mesh, multiples=(batch_size, 1, 1, 1))

        # lastly concatenate along the channel axis
        return tf.concat([z, x, y], axis=-1)

    def get_xy_meshgrid(self, w: int, h: int):
        x_coord = tf.linspace(-1, 1, w)
        y_coord = tf.linspace(-1, 1, h)

        # meshgrid & cast
        x, y = tf.meshgrid(x_coord, y_coord)
        x = tf.cast(x, dtype=tf.float32)
        y = tf.cast(y, dtype=tf.float32)

        # expand shape (to 4D) to later match the tiled latents
        x = tf.reshape(x, shape=self.tile_shape)
        y = tf.reshape(y, shape=self.tile_shape)
        return x, y

    def get_config(self) -> dict:
        config: dict = super().get_config()
        config.update(width=self.w, height=self.h)
        return config


class SpatialConditioning(tfkl.Layer):
    """A layer able to condition a volume activation (4D tensor out of conv layers) by a vector."""

    def __init__(self, size: tuple, filters: int, kernel: int, name=None):
        super().__init__(name=name)
        self.config = dict(size=size, kernel=kernel, filters=filters)

        self.w = size[1]
        self.h = size[0]
        self.z = None

        self.broadcast = SpatialBroadcast(width=self.w, height=self.h)
        self.expand = tfkl.Conv2D(filters=int(filters), kernel_size=kernel, activation='linear',
                                  padding='same')

    def build(self, input_shape: list):
        super().build(input_shape)

        _, latents_shape = input_shape
        self.z = latents_shape[-1]

    def call(self, inputs: list, **kwargs):
        x, z = inputs

        # conditioning representation
        z = self.broadcast(z)
        z.set_shape((None, self.h, self.w, self.z + 2))
        z = self.expand(z)

        # condition by scaling
        return x * z

    def get_config(self) -> dict:
        return self.config


class ConditionalBiasing2D(SpatialConditioning):
    """A layer able to condition a volume activation (4D tensor out of conv layers) by a vector."""

    def call(self, inputs: list, **kwargs):
        x, z = inputs

        # conditioning representation
        z = self.broadcast(z)
        z.set_shape((None, self.h, self.w, self.z + 2))
        z = self.expand(z)

        # condition by addition
        return x + z


class AffineConditioning2D(SpatialConditioning):
    def __init__(self, size: tuple, filters: int, kernel: int, name=None):
        super().__init__(size, filters, kernel, name=name)

        self.expand1 = self.expand
        self.expand2 = tfkl.Conv2D(filters=int(filters), kernel_size=kernel, activation='linear',
                                   padding='same')

    def call(self, inputs: list, **kwargs):
        x, z = inputs

        # conditioning representation
        z = self.broadcast(z)
        z.set_shape((None, self.h, self.w, self.z + 2))

        s = self.expand1(z)
        b = self.expand2(z)

        # condition by scaling and biasing (i.e. affine-like)
        return x * s + b


class AffineConditioning(tfkl.Layer):
    """Generalized affine transform-based conditioning layer"""

    def __init__(self, scale_activation='linear', bias_activation='linear', name=None, **kwargs):
        super().__init__(name=name)
        self.kwargs = kwargs

        self.scale_activation = scale_activation
        self.bias_activation = bias_activation

        self.dense_scale: tfkl.Dense = None
        self.dense_bias: tfkl.Dense = None

        self.multiply = tfkl.Multiply()
        self.add = tfkl.Add()

    def build(self, input_shape):
        shape, _ = input_shape

        self.dense_scale = tfkl.Dense(units=shape[-1], activation=self.scale_activation, **self.kwargs)
        self.dense_bias = tfkl.Dense(units=shape[-1], activation=self.bias_activation, **self.kwargs)

    def call(self, inputs, **kwargs):
        assert isinstance(inputs, (list, tuple))
        assert len(inputs) == 2

        # condition input `x` on `z`
        x, z = inputs

        scale = self.dense_scale(z)
        bias = self.dense_bias(z)

        # apply affine transformation, i.e. y = scale(z) * x + bias(z)
        y = self.multiply([x, scale])
        y = self.add([y, bias])
        return y


class ConvLayer(tfkl.Layer):
    def __init__(self, filters: int, kernel: int, stride=1, name=None,
                 activation=tf.nn.relu6, dropout=0.0, norm='instance', **kwargs):
        super().__init__(name=name)
        norm = norm.lower()

        self.config = dict(filters=filters, kernel=kernel, activation=activation)

        self.conv = tfkl.Conv2D(filters=filters, kernel_size=kernel, strides=stride,
                                padding='same', **kwargs)

        if norm == 'instance':
            self.norm = tfa.layers.InstanceNormalization()
        elif norm == 'group':
            self.norm = tfa.layers.GroupNormalization(groups=filters // 4)
        else:
            self.norm = tfkl.BatchNormalization()

        self.act = tfkl.Activation(activation)

        if dropout > 0.0:
            self.dropout = tfkl.SpatialDropout2D(rate=float(dropout))
        else:
            self.dropout = None

    @property
    def layers(self) -> List[tfkl.Layer]:
        return [self.conv, self.norm, self.act]

    def call(self, x, **kwargs):
        x = self.conv(x)
        x = self.norm(x, **kwargs)
        x = self.act(x)

        if self.dropout is not None:
            return self.dropout(x, **kwargs)

        return x

    def get_config(self) -> dict:
        return self.config


class UpConvLayer(tfkl.Layer):
    def __init__(self, filters: int, kernel: int, name=None, norm='instance', up_size=(2, 2),
                 activation=tf.nn.relu6, interp='bilinear', dropout=0.0, **kwargs):
        super().__init__(name=name)
        norm = norm.lower()
        self.config = dict(filters=filters, kernel=kernel, activation=activation)

        self.up_sample = tfkl.UpSampling2D(size=up_size, interpolation=interp)
        self.conv = tfkl.Conv2D(filters=filters, kernel_size=kernel,
                                padding='same', **kwargs)

        if norm == 'instance':
            self.norm = tfa.layers.InstanceNormalization()
        elif norm == 'group':
            self.norm = tfa.layers.GroupNormalization(groups=filters // 4)
        else:
            self.norm = tf.keras.layers.BatchNormalization()

        self.act = tfkl.Activation(activation)

        if dropout > 0.0:
            self.dropout = tfkl.SpatialDropout2D(rate=float(dropout))
        else:
            self.dropout = None

    @property
    def layers(self) -> List[tfkl.Layer]:
        return [self.up_sample, self.conv, self.norm, self.act]

    def call(self, x, **kwargs):
        x = self.up_sample(x)
        x = self.conv(x)
        x = self.norm(x, **kwargs)
        x = self.act(x)

        if self.dropout is not None:
            return self.dropout(x, **kwargs)

        return x

    def get_config(self) -> dict:
        return self.config


def get_normalization(name: str, layer_name=None) -> tf.keras.layers.Layer:
    if name.lower() == 'instance':
        return tfa.layers.InstanceNormalization(name=layer_name)

    return tf.keras.layers.BatchNormalization(name=layer_name)
