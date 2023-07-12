import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from ad import plot, utils


class PlotCallback(tf.keras.callbacks.Callback):
    """Custom callback that plots reconstructions"""

    def __init__(self, path: str, freq=1, amount=1, data: np.ndarray = None, normalize_fn=None,
                 predict_fn=None, post_fn=None, **kwargs):
        self.freq = int(freq)
        self.amount = int(amount)
        self.kwargs = kwargs

        # load samples from path
        if isinstance(path, str):
            self.data = utils.read_npz(path, dtype=np.float32, verbose=False, keys='images')
        else:
            assert isinstance(data, np.ndarray)
            self.data = data

        if callable(normalize_fn):
            self.data = normalize_fn(self.data)

        if callable(predict_fn):
            self.predict_fn = lambda model, x: predict_fn(model, x)
        else:
            self.predict_fn = lambda model, x: self.model(x)

        if callable(post_fn):
            self.post_process = lambda x: post_fn(x)
        else:
            self.post_process = lambda x: x

    def plot(self):
        x = self.data[np.random.choice(len(self.data), size=self.amount, replace=False)]
        y = self.predict_fn(self.model, x)
        x = self.post_process(x)

        if self.amount > 1:
            x = np.mean(x, axis=0)
            y = np.mean(y, axis=0)

        plot.compare(x, y, **self.kwargs)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.freq == 0:
            self.plot()


class PlotLatents(PlotCallback):
    def __init__(self, *args, predict_fn, bins=100, **kwargs):
        super().__init__(*args, **kwargs)
        assert callable(predict_fn)

        self.bins = int(bins)
        self.buffer = []
        self.predict = predict_fn

    def plot(self, z1, z2):
        ax1, ax2 = utils.get_plot_axes(rows=1, cols=2)

        # histogram
        ax1.hist(z1, bins=self.bins, density=True, histtype='step', label=r'z$_0$')
        ax1.hist(z2, bins=self.bins, density=True, histtype='step', label=r'$z_1$')
        ax1.legend(loc='best')

        # scatter plot
        ax2.scatter(z1, z2, marker='o', s=10, label=r'$z$')
        ax2.set_xlabel(r'$z_0$')
        ax2.set_ylabel(r'$z_1$')
        ax2.legend(loc='best')

        plt.show()

    def on_epoch_end(self, epoch, logs=None):
        x = self.data[np.random.choice(len(self.data), size=self.amount, replace=False)]
        z = self.predict(self.model, x)

        z1 = z[:, 0]
        z2 = z[:, 1]

        self.buffer.append((z1, z2))

        if epoch % self.freq == 0:
            self.plot(z1, z2)

    def save_latents(self, path: str, name: str, clear=True):
        for i, (z1, z2) in enumerate(self.buffer):
            plt.figure(figsize=(12, 10))

            plt.scatter(z1, z2, marker='o', s=10, label=r'$z$')
            plt.xlabel(r'$z_0$')
            plt.ylabel(r'$z_1$')
            plt.legend(loc='best')

            # save figure
            path = utils.makedir(path)
            plt.savefig(os.path.join(path, f'{name}_{i}.png'), bbox_inches='tight')

            plt.show()

        # empty buffer
        if clear:
            self.buffer.clear()
