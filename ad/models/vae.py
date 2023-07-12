import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import ad

from ad import utils
from ad.models import AE
from typing import List


tfd = tfp.distributions
tfl = tfp.layers
tfb = tfp.bijectors


class COD2VAE(AE):
    """Conditional Dual Categorical VAE (CoDVAE)"""

    def __init__(self, image_shape: tuple, latent_size: int, name=None, reshape=(5, 4),
                 kl_weight=1.0, mc_samples=10, tau=0.5, kl_reduce_axis=(1, 2), **kwargs):
        self._base_model_initialized = True

        self.latent_shape = reshape + (latent_size,)
        self.base_tau = tf.constant(float(tau), dtype=tf.float32)

        self.prior = self.get_prior(latent_shape=self.latent_shape)
        self.kl_reg = self.get_kl_regularizer(kl_weight=float(kl_weight), num_samples=int(mc_samples),
                                              reduction_axis=kl_reduce_axis)
        # models
        self.mask_encoder = self.get_mask_encoder(input_shape=image_shape,
                                                  **kwargs.get('mask_encoder', {}))

        self.energy_encoder = self.get_energy_encoder(input_shape=image_shape,
                                                      **kwargs.pop('large_encoder', {}))

        self.decoder = self.get_decoder(out_channels=1, crop=image_shape[:2],
                                        **kwargs.pop('decoder', {}))

        # metrics
        self.trackers = dict(loss=tf.keras.metrics.Mean(name='total_loss'),
                             mse=ad.metrics.MSE(name='mse'),
                             psnr=ad.metrics.PSNR(name='psnr'),
                             ssim=ad.metrics.SSIM(name='ssim'),
                             tau=tf.keras.metrics.Mean(name='temperature'),
                             kl_loss=tf.keras.metrics.Mean(name='kl_loss'),
                             reco_loss=tf.keras.metrics.Mean(name='reconstruction_loss'),
                             true_energy=tf.keras.metrics.Mean(name='true_energy'),
                             pred_energy=tf.keras.metrics.Mean(name='pred_energy'),
                             grads_norm=tf.keras.metrics.Mean(name='gradients_norm'),
                             weights_norm=tf.keras.metrics.Mean(name='weights_norm'))

        self.test_trackers = [k for k in self.trackers.keys() if '_norm' not in k]

        super().__init__(name=name)

    def call(self, x: tf.Tensor, training=None):
        zm = self.mask_encoder(x, training=training)
        ze, _ = self.energy_encoder(x, training=training)

        dist = self.decoder([ze, zm], training=training)

        if not training:
            return dist.mode()

        return dist

    @tf.function
    def train_step(self, batch: tf.Tensor):
        x_mask = self._prepare_data(batch)

        with tf.GradientTape() as tape:
            zm = self.mask_encoder(x_mask, training=True)
            ze, tau = self.energy_encoder(x_mask, training=True)
            bern = self.decoder([ze, zm], training=True)

            # losses
            reco_loss = self.reconstruction_loss(images=x_mask, decoding_distribution=bern)
            kl_loss = tf.reduce_sum(self.energy_encoder.losses)
            total_loss = reco_loss + kl_loss

        weights = self.trainable_variables
        grads = tape.gradient(total_loss, weights)
        self.optimizer.apply_gradients(zip(grads, weights))

        # reconstructions
        x = tf.identity(bern)

        self.update_trackers(loss=total_loss, mse=(x, x_mask), psnr=(x, x_mask),
                             reco_loss=reco_loss, ssim=(x, x_mask),
                             kl_loss=kl_loss, tau=tau,
                             true_energy=tf.reduce_sum(x_mask, axis=[1, 2, 3]),
                             pred_energy=tf.reduce_sum(x, axis=[1, 2, 3]),
                             grads_norm=utils.tf_global_norm(grads),
                             weights_norm=utils.tf_global_norm(weights))

        return {k: metric.result() for k, metric in self.trackers.items()}

    @tf.function
    def test_step(self, batch: tf.Tensor):
        x_mask = self._prepare_data(batch, augment=False)

        zm = self.mask_encoder(x_mask, training=False)
        ze, tau = self.energy_encoder(x_mask, training=False)
        bern = self.decoder([ze, zm], training=False)

        # losses
        reco_loss = self.reconstruction_loss(images=x_mask, decoding_distribution=bern)
        kl_loss = tf.reduce_sum(self.energy_encoder.losses)
        total_loss = reco_loss + kl_loss

        # reconstructions
        x = tf.identity(bern.mode())

        self.update_trackers(loss=total_loss, mse=(x, x_mask), psnr=(x, x_mask),
                             reco_loss=reco_loss, ssim=(x, x_mask),
                             kl_loss=kl_loss, tau=tau,
                             true_energy=tf.reduce_sum(x_mask, axis=[1, 2, 3]),
                             pred_energy=tf.reduce_sum(x, axis=[1, 2, 3]))

        return {k: self.trackers[k].result() for k in self.test_trackers}

    @classmethod
    def get_prior(cls, latent_shape: tuple, seed=utils.SEED, width=3):
        # latent_shape = (A, B, C) -> reshape to (B, A, C)
        latent_shape = (latent_shape[1], latent_shape[0], latent_shape[2])
        num_modes = latent_shape[1]  # A

        mixture_probs = np.ones(shape=(num_modes,), dtype=np.float32) / num_modes
        shape = (num_modes, np.prod(latent_shape))
        means = np.linspace(-width, width, num=num_modes)

        taus = tf.random.uniform(shape=(num_modes,), minval=0.1, maxval=1.0, seed=seed)
        logits = tf.random.normal(shape, seed=seed) + means.reshape((-1, 1))
        logits = tf.Variable(tf.reshape(logits, shape=(num_modes,) + latent_shape))

        # final event shape = (A, B, C)
        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=mixture_probs),
            components_distribution=tfd.RelaxedOneHotCategorical(temperature=taus, logits=logits))

    def get_kl_regularizer(self, kl_weight=1.0, num_samples=3, reduction_axis=(1, 2)):
        return tfl.KLDivergenceRegularizer(self.prior, use_exact_kl=False, weight=float(kl_weight),
                                           test_points_fn=lambda q: q.sample(num_samples),
                                           test_points_reduce_axis=reduction_axis)

    def get_energy_encoder(self, input_shape: tuple, depths: List[int], filters: List[int],
                           activation=tf.nn.leaky_relu, kernel=3, groups=None,
                           **kwargs) -> tf.keras.Model:
        from tensorflow.keras.layers import Input, Dense, Flatten, Add, Conv2D, Reshape
        assert len(depths) == len(filters)

        images = Input(shape=input_shape, name='energy_image')
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
        logits = Conv2D(filters=self.latent_shape[-1], kernel_size=1, padding='same',
                        name='logits')(x)

        # temperature
        tau = Conv2D(filters=1, activation=tf.nn.softplus, name='temperature',
                     padding='same', kernel_size=1)(x) + self.base_tau
        tau = Reshape(tau.shape[1:3], name='reshape')(tau)

        z = tfl.DistributionLambda(self.get_distribution, name='gumbel_softmax',
                                   activity_regularizer=self.kl_reg)([tau, logits])

        return tf.keras.Model(inputs=images, outputs=[z, tau], name='Energy-Encoder')

    def get_mask_encoder(self, input_shape: tuple, filters: List[int], kernel=3,
                         activation=tf.nn.leaky_relu, groups=None, **kwargs):
        from tensorflow.keras.layers import Input, Dense, MaxPool2D, Flatten

        images = Input(shape=input_shape, name='mask_image')
        x = images

        for i, num_filters in enumerate(filters):
            x = ad.layers.ConvLayer(filters=int(num_filters), kernel=kernel, **kwargs,
                                    groups=groups if i > 0 else None, activation=activation,
                                    name=f'conv-b{i}')(x)

            x = MaxPool2D(pool_size=3, strides=2, padding='same', name=f'max_pool-b{i}')(x)

        x = Flatten(name='flatten')(x)
        z = Dense(units=2, name='z')(x)

        return tf.keras.Model(inputs=images, outputs=z, name='Mask-Encoder')

    @staticmethod
    def get_distribution(params: list):
        temp, logits = params

        return tfd.RelaxedOneHotCategorical(temperature=temp, logits=logits)

    def get_decoder(self, depths: List[int], filters: List[int], crop: tuple,
                    kernel=3, activation=tf.nn.leaky_relu, out_channels=1,
                    out_kernel=1, groups=None, bias=0.0, **kwargs) -> tf.keras.Model:
        from tensorflow.keras.layers import Input, Add, Reshape, CenterCrop, Conv2D, Flatten, \
                                            Concatenate
        assert len(depths) == len(filters)

        ze = Input(shape=self.latent_shape, name='energy_latents')
        zm = Input(shape=2, name='mask_latents')

        x = ze

        for j, depth in enumerate(depths):
            x = ad.layers.SpatialConditioning(size=x.shape[1:3], filters=x.shape[-1],
                                              kernel=kernel, name=f'conditioning-b{j}')([x, zm])

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

        x = Conv2D(filters=out_channels, kernel_size=out_kernel, padding='same',
                   bias_initializer=tf.keras.initializers.Constant(float(bias)), **kwargs)(x)
        x = Flatten()(x)

        reco = tfl.IndependentBernoulli(event_shape=crop + (out_channels,), name='bern')(x)

        return tf.keras.Model(inputs=[ze, zm], outputs=reco, name='Res-Decoder')

    @staticmethod
    def reconstruction_loss(images: tf.Tensor, decoding_distribution: tfd.Distribution):
        return -tf.reduce_mean(decoding_distribution.log_prob(images), axis=0)

    def summary(self, **kwargs):
        self.mask_encoder.summary(**kwargs)
        self.energy_encoder.summary(**kwargs)
        self.decoder.summary(**kwargs)
