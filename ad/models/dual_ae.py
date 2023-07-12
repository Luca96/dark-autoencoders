import tensorflow as tf

import ad

from ad import utils
from ad.models import AE
from typing import List


class CoDAE(AE):
    """Conditional Dual Auto-Encoder (CoDAE)"""

    def __init__(self, image_shape: tuple, latent_size: int, name=None, **kwargs):
        self._base_model_initialized = True

        self.latent_size = int(latent_size)

        # models
        self.encoder1 = self.get_encoder1(input_shape=image_shape,
                                          **kwargs.pop('encoder1', {}))

        self.encoder2 = self.get_encoder2(input_shape=image_shape,
                                          **kwargs.get('encoder2', {}))

        self.decoder = self.get_decoder(out_channels=image_shape[-1], crop=image_shape[:2],
                                        latent_shape=self.encoder1.output.shape[1:],
                                        **kwargs.pop('decoder', {}))

        # metrics
        self.trackers = dict(loss=tf.keras.metrics.Mean(name='total_loss'),
                             mse=ad.metrics.MSE(name='mse'),
                             psnr=ad.metrics.PSNR(name='psnr'),
                             ssim=ad.metrics.SSIM(name='ssim'),
                             true_energy=tf.keras.metrics.Mean(name='true_energy'),
                             pred_energy=tf.keras.metrics.Mean(name='pred_energy'),
                             grads_norm=tf.keras.metrics.Mean(name='gradients_norm'),
                             weights_norm=tf.keras.metrics.Mean(name='weights_norm'))

        self.test_trackers = [k for k in self.trackers.keys() if '_norm' not in k]

        super().__init__(name=name)

    def call(self, x: tf.Tensor, training=None):
        z1 = self.encoder1(x, training=training)
        z2 = self.encoder2(x, training=training)

        return self.decoder([z1, z2], training=training)

    @tf.function
    def train_step(self, batch: tf.Tensor):
        x1 = self._prepare_data(batch)

        with tf.GradientTape() as tape:
            z1 = self.encoder1(x1, training=True)
            z2 = self.encoder2(x1, training=True)
            x = self.decoder([z1, z2], training=True)

            # losses
            total_loss = self.compiled_loss(x, x1)

        weights = self.trainable_variables
        grads = tape.gradient(total_loss, weights)
        self.optimizer.apply_gradients(zip(grads, weights))

        self.update_trackers(loss=total_loss, mse=(x, x1), psnr=(x, x1),
                             ssim=(x, x1),
                             true_energy=tf.reduce_sum(x1, axis=[1, 2, 3]),
                             pred_energy=tf.reduce_sum(x, axis=[1, 2, 3]),
                             grads_norm=utils.tf_global_norm(grads),
                             weights_norm=utils.tf_global_norm(weights))

        return {k: metric.result() for k, metric in self.trackers.items()}

    @tf.function
    def test_step(self, batch: tf.Tensor):
        z1 = self.encoder1(batch, training=False)
        z2 = self.encoder2(batch, training=False)
        x = self.decoder([z1, z2], training=False)

        # losses
        total_loss = self.compiled_loss(x, batch)

        self.update_trackers(loss=total_loss, mse=(x, batch), psnr=(x, batch),
                             ssim=(x, batch),
                             true_energy=tf.reduce_sum(batch, axis=[1, 2, 3]),
                             pred_energy=tf.reduce_sum(x, axis=[1, 2, 3]))

        return {k: self.trackers[k].result() for k in self.test_trackers}

    def get_encoder1(self, input_shape: tuple, depths: List[int], filters: List[int],
                     activation=tf.nn.leaky_relu, kernel=3, groups=None,
                     **kwargs) -> tf.keras.Model:
        from tensorflow.keras.layers import Input, Add, Conv2D
        assert len(depths) == len(filters)

        images = Input(shape=input_shape, name='image')
        x = images

        for j, depth in enumerate(depths):
            x = ad.layers.ConvLayer(filters=filters[j], kernel=kernel, stride=2, **kwargs,
                                    groups=groups if j > 0 else None,
                                    activation=activation, name=f'dconv-b{j}')(x)

            # add residual blocks
            for i in range(depth):
                r = x  # residual

                x = ad.layers.ConvLayer(filters=filters[j], kernel=kernel, **kwargs,
                                        groups=groups,
                                        activation=activation, name=f'conv1-b{j}_{i}')(x)
                x = ad.layers.ConvLayer(filters=filters[j], kernel=kernel, **kwargs,
                                        groups=groups,
                                        activation=activation, name=f'conv2-b{j}_{i}')(x)

                x = Add(name=f'add-b{j}_{i}')([x, r])

        # latent space
        z = Conv2D(filters=self.latent_size, kernel_size=1, padding='same',
                   name='latents')(x)

        return tf.keras.Model(inputs=images, outputs=z, name='Res-Encoder')

    def get_encoder2(self, input_shape: tuple, filters: List[int], kernel=3,
                     activation=tf.nn.leaky_relu, groups=None, **kwargs):
        from tensorflow.keras.layers import Input, Dense, MaxPool2D, Flatten

        images = Input(shape=input_shape, name='image')
        x = images

        for i, num_filters in enumerate(filters):
            x = ad.layers.ConvLayer(filters=int(num_filters), kernel=kernel, **kwargs,
                                    groups=groups if i > 0 else None, activation=activation,
                                    name=f'conv-b{i}')(x)

            x = MaxPool2D(pool_size=3, strides=2, padding='same', name=f'max_pool-b{i}')(x)

        x = Flatten(name='flatten')(x)
        z = Dense(units=2, name='z')(x)

        return tf.keras.Model(inputs=images, outputs=z, name='Mask-Encoder')

    def get_decoder(self, depths: List[int], filters: List[int], crop: tuple,
                    latent_shape: tuple, kernel=3, activation=tf.nn.leaky_relu,
                    out_channels=1, out_kernel=1, groups=None, bias=0.0,
                    **kwargs) -> tf.keras.Model:
        from tensorflow.keras.layers import Input, Add, Reshape, CenterCrop, Conv2D, Flatten, \
                                            Concatenate
        assert len(depths) == len(filters)

        z1 = Input(shape=latent_shape, name='energy_latents')
        z2 = Input(shape=2, name='mask_latents')

        x = z1

        for j, depth in enumerate(depths):
            x = ad.layers.SpatialConditioning(size=x.shape[1:3], filters=x.shape[-1],
                                              kernel=kernel, name=f'conditioning-b{j}')([x, z2])

            x = ad.layers.UpConvLayer(filters=filters[j], kernel=kernel, **kwargs,
                                      groups=groups if j > 0 else None,
                                      activation=activation, name=f'up_conv-b{j}')(x)
            # add residual blocks
            for i in range(depth):
                r = x  # residual

                x = ad.layers.ConvLayer(filters=filters[j], kernel=kernel, **kwargs,
                                        groups=groups,
                                        activation=activation, name=f'conv1-b{j}_{i}')(x)
                x = ad.layers.ConvLayer(filters=filters[j], kernel=kernel, **kwargs,
                                        groups=groups,
                                        activation=activation, name=f'conv2-b{j}_{i}')(x)

                x = Add(name=f'add-b{j}_{i}')([x, r])

        # reconstruction
        x = CenterCrop(*crop, name='crop')(x)

        reco = Conv2D(filters=out_channels, kernel_size=out_kernel, padding='same',
                      bias_initializer=tf.keras.initializers.Constant(float(bias)),
                      activation=tf.nn.sigmoid, **kwargs)(x)

        return tf.keras.Model(inputs=[z1, z2], outputs=reco, name='Res-Decoder')

    def summary(self, **kwargs):
        self.encoder1.summary(**kwargs)
        self.encoder2.summary(**kwargs)
        self.decoder.summary(**kwargs)
