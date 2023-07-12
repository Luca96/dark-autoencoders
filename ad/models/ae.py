
import ad.constants
import tensorflow as tf

from ad import utils
from functools import partial


class AE(tf.keras.Model):
    """Base class for auto-encoders"""

    @classmethod
    def bce_loss(cls, y_pred, y_true):
        bce = tf.keras.losses.BinaryCrossentropy(axis=[], reduction='none')

        loss = bce(y_true, y_pred)
        loss = tf.reduce_sum(loss, axis=[1, 2, 3])
        return tf.reduce_mean(loss)

    @classmethod
    def mse_loss(cls, y_pred, y_true):
        loss = tf.square(y_true - y_pred)
        loss = tf.reduce_sum(loss, axis=[1, 2, 3])
        return tf.reduce_mean(loss)

    @tf.function
    def dice_loss(self, y_pred, y_true):
        # Source: https://arxiv.org/pdf/1807.10097v1.pdf (page 6)
        sum_p = tf.reduce_sum(tf.square(y_pred), axis=[1, 2, 3])
        sum_t = tf.reduce_sum(tf.square(y_true), axis=[1, 2, 3])

        union = sum_p + sum_t
        intersection = tf.reduce_sum(y_pred * y_true, axis=[1, 2, 3])

        loss = union / (2.0 * intersection)
        return tf.reduce_sum(loss)

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

    @tf.function
    def augment(self, x, seed: int):
        return tf.map_fn(fn=partial(ad.aug.tf_augment, seed=seed),
                         elems=x, parallel_iterations=16)

    @tf.function
    def _prepare_data(self, batch: tf.Tensor, augment=True):
        if augment:
            return self.augment(batch, seed=utils.SEED)

        return batch
