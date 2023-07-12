"""Models from other papers"""
import keras.metrics
import numpy as np
import tensorflow as tf

import ad

from ad import utils
from typing import List, Tuple
from tensorflow.keras import layers as tfkl


class HeimelAE(tf.keras.Model):
    """AE inspired by 'QCD or What?' (10.21468/SciPostPhys.6.3.030)"""

    def __init__(self, image_shape: tuple, latent_size: int, **kwargs):
        self._base_model_initialized = True
        self.latent_size = int(latent_size)

        # models
        self.encoder = self.get_encoder(input_shape=image_shape, **kwargs.get('encoder', {}))
        self.decoder = self.get_decoder(**kwargs.get('decoder', {}))

        # metrics
        self.trackers = dict(loss=tf.keras.metrics.Mean(name='total_loss'),
                             mse=ad.metrics.MSE(name='mse'),
                             ssim=ad.metrics.SSIM(name='ssim'),
                             true_energy=tf.keras.metrics.Mean(name='true_energy'),
                             pred_energy=tf.keras.metrics.Mean(name='pred_energy'),
                             grads_norm=tf.keras.metrics.Mean(name='gradients_norm'),
                             weights_norm=tf.keras.metrics.Mean(name='weights_norm'))

        self.test_trackers = [k for k in self.trackers.keys() if '_norm' not in k]

        super().__init__()

    def call(self, x: tf.Tensor, training=None):
        z = self.encoder(x, training=training)
        return self.decoder(z, training=training)

    @tf.function
    def train_step(self, batch: tf.Tensor):
        batch = self.augment(batch)

        with tf.GradientTape() as tape:
            z = self.encoder(batch, training=True)
            x = self.decoder(z, training=True)

            # losses
            total_loss = self.compiled_loss(x, batch)

        weights = self.trainable_variables
        grads = tape.gradient(total_loss, weights)
        self.optimizer.apply_gradients(zip(grads, weights))

        self.update_trackers(loss=total_loss, mse=(x, batch), ssim=(x, batch),
                             true_energy=tf.reduce_sum(batch, axis=[1, 2, 3]),
                             pred_energy=tf.reduce_sum(x, axis=[1, 2, 3]),
                             grads_norm=utils.tf_global_norm(grads),
                             weights_norm=utils.tf_global_norm(weights))

        return {k: metric.result() for k, metric in self.trackers.items()}

    @tf.function
    def test_step(self, batch: tf.Tensor):
        z = self.encoder(batch, training=False)
        x = self.decoder(z, training=False)

        # losses
        total_loss = self.compiled_loss(x, batch)

        self.update_trackers(loss=total_loss, mse=(x, batch), ssim=(x, batch),
                             true_energy=tf.reduce_sum(batch, axis=[1, 2, 3]),
                             pred_energy=tf.reduce_sum(x, axis=[1, 2, 3]))

        return {k: self.trackers[k].result() for k in self.test_trackers}

    def get_encoder(self, input_shape: tuple, filters: List[Tuple[int, int]],
                    units: List[int], **kwargs):
        image = tfkl.Input(shape=input_shape, name='image')
        x = image

        for i, (f1, f2) in enumerate(filters):
            # conv-1
            x = tfkl.Conv2D(filters=f1, kernel_size=3, activation=None, padding='same',
                            name=f'conv1-{i}', **kwargs)(x)
            x = tfkl.PReLU(shared_axes=[1, 2], name=f'PReLU1-{i}')(x)

            # conv-2
            if f2 is not None:
                x = tfkl.Conv2D(filters=f2, kernel_size=3, activation=None, padding='same',
                                name=f'conv2-{i}', **kwargs)(x)
                x = tfkl.PReLU(shared_axes=[1, 2], name=f'PReLU2-{i}')(x)

            # pool
            x = tfkl.AvgPool2D(name=f'avg_pool-{i}')(x)

        x = tfkl.Flatten(name='flatten')(x)

        for i, num_units in enumerate(units):
            x = tfkl.Dense(num_units, activation=None, **kwargs, name=f'dense-{i}')(x)
            x = tfkl.PReLU(name=f'PReLU-{i}')(x)

        z = tfkl.Dense(units=self.latent_size, name='latents', **kwargs)(x)
        return tf.keras.Model(image, z, name='Encoder')

    def get_decoder(self, filters: List[Tuple[int, int]], units: List[int],
                    reshape_to: tuple, crop=(72, 58), bias=0.0, **kwargs):
        z = tfkl.Input(shape=self.latent_size, name='z')
        x = z

        for i, num_units in enumerate(units):
            x = tfkl.Dense(num_units, name=f'dense-{i}', **kwargs)(x)
            x = tfkl.PReLU(name=f'PReLU-{i}')(x)

        x = tfkl.Reshape(reshape_to, name='reshape')(x)

        for i, (f1, f2) in enumerate(filters):
            x = tfkl.Conv2DTranspose(filters=f1, kernel_size=3, padding='same', **kwargs,
                                     strides=2, name=f'conv_t1-{i}')(x)
            x = tfkl.PReLU(shared_axes=[1, 2], name=f'PReLU1-{i}')(x)

            x = tfkl.Conv2D(filters=f1, kernel_size=3, padding='same', **kwargs,
                            name=f'conv_2-{i}')(x)
            x = tfkl.PReLU(shared_axes=[1, 2], name=f'PReLU2-{i}')(x)

        if isinstance(crop, tuple):
            x = tfkl.CenterCrop(*crop, name='crop')(x)

        reco = tfkl.Conv2D(filters=1, kernel_size=1, activation=tf.nn.sigmoid, padding='same',
                           bias_initializer=tf.keras.initializers.Constant(bias),
                           **kwargs)(x)

        return tf.keras.Model(z, reco, name='Decoder')

    @classmethod
    def mse_loss(cls, true, pred):
        loss = tf.reduce_sum(tf.square(true - pred), axis=[1, 2, 3])
        return tf.reduce_mean(loss)

    @tf.function
    def augment(self, x: tf.Tensor):
        return tf.map_fn(fn=ad.aug.tf_augment, elems=x, parallel_iterations=16)

    def update_trackers(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.trackers:
                continue

            if isinstance(v, tuple):
                self.trackers[k].update_state(*v)

            elif isinstance(v, dict):
                self.trackers[k].update_state(**v)
            else:
                self.trackers[k].update_state(v)

    def summary(self, **kwargs):
        self.encoder.summary(**kwargs)
        self.decoder.summary(**kwargs)


class GaussianEncoder(tf.keras.Model):
    def __init__(self, input_size: int, prior_mean: np.ndarray, prior_var: np.ndarray,
                 units=512, latent_size=2, name=None, dtype=tf.float32, **kwargs):
        assert latent_size >= 1

        self.latent_size = int(latent_size)
        self.prior_mean = tf.constant(prior_mean, dtype=dtype)
        self.prior_var = tf.constant(prior_var, dtype=dtype)
        self.datatype = dtype

        image = tfkl.Input(shape=input_size, name='images')
        x = image

        x = tfkl.Dense(units=units, activation='selu', dtype=dtype, **kwargs)(x)
        x = tfkl.BatchNormalization(center=False, scale=True)(x)

        mean = tfkl.Dense(units=self.latent_size, activation='linear')(x)
        log_var = tfkl.Dense(units=self.latent_size, activation='linear')(x)

        super().__init__(inputs=image, outputs=[mean, log_var], name=name)

    def sample(self, mean, log_var):
        batch_size = tf.shape(mean)[0]
        eps = tf.random.normal(shape=(batch_size, self.latent_size), dtype=self.datatype)

        return mean + tf.math.exp(0.5 * log_var) * eps

    def call(self, inputs, **kwargs):
        z_mean, z_log_var = super().call(inputs, **kwargs)

        # clip log_var for stability
        z_log_var = tf.clip_by_value(z_log_var, -20.0, 2.0)

        return z_mean, z_log_var, self.sample(z_mean, z_log_var)


class DirichletDecoder(tf.keras.Model):
    def __init__(self, input_size: int, latent_size=2, activation=tf.nn.softmax, name=None, **kwargs):
        self.latent_size = int(latent_size)

        z = tfkl.Input(shape=latent_size, name='latents')
        out = tfkl.Dense(units=int(input_size), activation='linear', use_bias=False, **kwargs)(z)

        self.out_activation = tf.keras.activations.get(activation)
        super().__init__(inputs=z, outputs=out, name=name)

    def call(self, z, **kwargs):
        # softmax-Gaussian approximation of the Dirichlet
        z_norm = tf.nn.softmax(z)
        h = super().call(z_norm, **kwargs)
        return self.out_activation(h)


class AUC(tf.keras.metrics.AUC):
    def __init__(self, name=None):
        super().__init__(from_logits=True, dtype=tf.float32, name=name)


class DirichletVAE(tf.keras.Model):
    """Better latent spaces for better autoencoders (10.21468/SciPostPhys.11.3.061)
       - Based on: https://github.com/bmdillon/jet-mixture-vae
    """
    def __init__(self, input_shape: tuple, concentration: np.ndarray, alpha=0.1,
                 name=None, dtype=tf.float32, out_activation=tf.nn.softmax, **kwargs):
        super().__init__(name=name)

        self.latent_size = len(concentration)
        input_size = int(np.prod(input_shape))
        prior_mean, prior_var = self.get_prior_params(concentration, dtype=dtype.as_numpy_dtype)

        print('prior mean:', prior_mean)
        print('prior variance:', prior_var)

        self.encoder = GaussianEncoder(input_size, prior_mean, prior_var, latent_size=self.latent_size,
                                       dtype=dtype, **kwargs)
        self.decoder = DirichletDecoder(input_size, latent_size=self.latent_size,
                                        activation=out_activation, **kwargs)
        self.reshape = tfkl.Reshape(input_shape)

        self.alpha = tf.constant(float(alpha), dtype=dtype)
        self.kl_loss = tf.keras.losses.KLDivergence()

        # metrics
        self.trackers = dict(kl_loss=tf.keras.metrics.Mean(name='kld'),
                             latent_loss=tf.keras.metrics.Mean(name='latent_loss'),
                             loss=tf.keras.metrics.Mean(name='total_loss'),
                             auc_z=AUC(name='latent_auc'),
                             mean=tf.keras.metrics.Mean(name='z_mean'),
                             var=tf.keras.metrics.Mean(name='z_variance'),
                             bkg_w=tf.keras.metrics.Mean(name='bkg_topic_weights'),
                             sig_w=tf.keras.metrics.Mean(name='sig_topic_weights'),
                             grads_norm=tf.keras.metrics.Mean(name='gradients_norm'),
                             weights_norm=tf.keras.metrics.Mean(name='weights_norm'))

        self.test_trackers = [k for k in self.trackers.keys() if '_norm' not in k]

    @property
    def metrics(self) -> list:
        return list(self.trackers.values())

    def call(self, inputs, training=False):
        mean, log_var, z = self.encoder(inputs, training=training)

        if training:
            reco = self.decoder(z)
        else:
            reco = self.decoder(mean)

        reco = self.reshape(reco)
        return mean, log_var, reco

    @tf.function
    def train_step(self, batch: tuple):
        x, y = batch

        with tf.GradientTape() as tape:
            mean, log_var, z = self.encoder(x, training=True)
            reco = self.decoder(z, training=True)

            kl_loss = self.kl_loss(x, reco)
            latent_loss = self.latent_loss(mean, log_var)
            total_loss = kl_loss + self.alpha * latent_loss

        grads_norm, weights_norm = self.apply_gradients(total_loss, tape)

        # compute metrics + AUCs
        topics_weights = tf.nn.softmax(mean)

        b_mask = tf.cast(y == 0.0, dtype=topics_weights.dtype)[:, tf.newaxis]
        s_mask = tf.cast(y == 1.0, dtype=topics_weights.dtype)[:, tf.newaxis]

        bkg_weights = tf.boolean_mask(tensor=(topics_weights * b_mask)[:, 0, tf.newaxis], mask=b_mask)
        sig_weights = tf.boolean_mask(tensor=(topics_weights * s_mask)[:, 0, tf.newaxis], mask=s_mask)

        self.update_trackers(loss=total_loss, kl_loss=kl_loss, latent_loss=latent_loss,
                             auc_z=(y, topics_weights[:, 0]), mean=mean, var=tf.exp(log_var),
                             bkg_w=bkg_weights, sig_w=sig_weights,
                             grads_norm=grads_norm, weights_norm=weights_norm)

        return {k: metric.result() for k, metric in self.trackers.items()}

    @tf.function
    def test_step(self, batch: tuple):
        x, y = batch

        mean, log_var, z = self.encoder(x, training=False)
        reco = self.decoder(mean, training=False)

        kl_loss = self.kl_loss(x, reco)
        latent_loss = self.latent_loss(mean, log_var)
        total_loss = kl_loss + self.alpha * latent_loss

        # compute metrics + AUCs
        topics_weights = tf.nn.softmax(mean)

        b_mask = tf.cast(y == 0.0, dtype=topics_weights.dtype)[:, tf.newaxis]
        s_mask = tf.cast(y == 1.0, dtype=topics_weights.dtype)[:, tf.newaxis]

        bkg_weights = tf.boolean_mask(tensor=(topics_weights * b_mask)[:, 0, tf.newaxis], mask=b_mask)
        sig_weights = tf.boolean_mask(tensor=(topics_weights * s_mask)[:, 0, tf.newaxis], mask=s_mask)

        self.update_trackers(loss=total_loss, kl_loss=kl_loss, latent_loss=latent_loss,
                             auc_z=(y, topics_weights[:, 0]), mean=mean, var=tf.exp(log_var),
                             bkg_w=bkg_weights, sig_w=sig_weights)

        return {k: self.trackers[k].result() for k in self.test_trackers}

    @tf.function
    def latent_loss(self, mean, log_var):
        sigma = tf.exp(log_var)
        batch_size = tf.cast(len(mean), dtype=mean.dtype)

        term1 = tf.reduce_sum(sigma / self.encoder.prior_var) / batch_size
        term2 = tf.reduce_sum(tf.multiply((self.encoder.prior_mean - mean) / self.encoder.prior_var,
                                          self.encoder.prior_mean - mean)) / batch_size
        term3 = tf.cast(self.latent_size, dtype=mean.dtype)
        term4 = tf.reduce_sum(tf.math.log(self.encoder.prior_var))
        term5 = tf.reduce_sum(log_var) / batch_size

        loss = term1 + term2 - term3 + term4 - term5
        return 0.5 * loss

    def update_trackers(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, tuple):
                self.trackers[k].update_state(*v)

            elif isinstance(v, dict):
                self.trackers[k].update_state(**v)
            else:
                self.trackers[k].update_state(v)

    def apply_gradients(self, loss, tape: tf.GradientTape):
        # compute the gradients of the `total_loss` w.r.t. the networks parameters
        grads = tape.gradient(loss, self.trainable_variables)
        trainable_vars = self.trainable_variables

        # take a gradient step that updates the weights
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        return utils.tf_global_norm(grads), utils.tf_global_norm(trainable_vars)

    @staticmethod
    def get_prior_params(alphas: np.ndarray, dtype=np.float64) -> tuple:
        """Computes the parameters (mean and variance) for the multivariate prior: a softmax-Gaussian."""
        a = np.reshape(alphas, newshape=(1, -1)).astype(dtype)
        # z_size = len(a)
        z_size = a.shape[-1]

        # prior mean
        mean = np.log(a).T - np.mean(np.log(a), axis=1)

        # prior variance
        a1 = 1.0 / a
        variance = (a1 * (1.0 - (2.0 / z_size))).T + (1.0 / (z_size * z_size)) * np.sum(a1)

        return mean.T, variance.T

    def summary(self, **kwargs):
        self.encoder.summary(**kwargs)
        self.decoder.summary(**kwargs)
