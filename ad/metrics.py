import numpy as np
import tensorflow as tf


class SSIM(tf.keras.metrics.MeanMetricWrapper):

    def __init__(self, image_range=1.0, **kwargs):
        super().__init__(lambda x, y: tf.image.ssim(x, y, max_val=float(image_range)), **kwargs)


class PSNR(tf.keras.metrics.MeanMetricWrapper):

    def __init__(self, image_range=1.0, **kwargs):
        super().__init__(lambda x, y: tf.image.psnr(x, y, max_val=float(image_range)), **kwargs)


class EnergySumMetric(tf.keras.metrics.MeanMetricWrapper):

    def __init__(self, stats: np.ndarray, **kwargs):
        # assert len(stats) == 3
        self.stats = tf.constant(stats, dtype=tf.float32)

        super().__init__(fn=self.sum_energy, dtype=tf.float32, **kwargs)

    @tf.function
    def sum_energy(self, x: tf.Tensor, y: tf.Tensor):
        energy = tf.reduce_sum(x * self.stats, axis=[1, 2, 3])
        return tf.reduce_mean(energy)

    def update_state(self, x, *args, **kwargs):
        return super().update_state(x, x, *args, **kwargs)


class MSE(tf.keras.metrics.MeanMetricWrapper):

    def __init__(self, **kwargs):
        super().__init__(fn=self.compute, dtype=tf.float32, **kwargs)

    @staticmethod
    @tf.function
    def compute(x: tf.Tensor, y: tf.Tensor):
        """MSE computed for images, so sum over spatial dimensions and average over batch"""
        sq_err = tf.square(x - y)
        error = tf.reduce_sum(sq_err, axis=[1, 2])
        return tf.reduce_mean(error)


# -------------------------------------------------------------------------------------------------
# -- Classification metrics compatible with `from_logits=True`
# -- See: https://github.com/tensorflow/tensorflow/issues/42182
# -------------------------------------------------------------------------------------------------

class MetricWrapper:
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.from_logits:
            if tf.rank(y_pred) == 1 or tf.shape(y_pred)[-1] == 1:
                y_pred = tf.nn.sigmoid(y_pred)
            else:
                y_pred = tf.nn.softmax(y_pred)

        return super().update_state(y_true, y_pred, sample_weight)


class AUC(MetricWrapper, tf.metrics.AUC):
    pass


class BinaryAccuracy(MetricWrapper, tf.metrics.BinaryAccuracy):
    pass


class CategoricalAccuracy(MetricWrapper, tf.metrics.CategoricalAccuracy):
    pass


class Precision(MetricWrapper, tf.metrics.Precision):
    pass


class Recall(MetricWrapper, tf.metrics.Recall):
    pass
