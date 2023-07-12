
import os
import numpy as np
import tensorflow as tf

import ad
from ad import utils

from typing import Union


def load_data(path: str, keys=None, dtype=np.float32, verbose=False, mask=True, **kwargs):
    if keys is None:
        keys = ['n_tracks', 'images', 'labels', 'masses']

    data = utils.read_npz(folder=path, keys=keys, dtype=dtype, verbose=verbose, **kwargs)

    if mask:
        x = (data['images'] > 0.0)[..., 0, np.newaxis].astype(dtype)

        return x, data['labels'], data['masses']

    return data


def profile(model: tf.keras.Model, data: np.ndarray, profiler_path: str, device='cpu'):
    """Use the tf Profiler to measure inference time of the provided model"""
    # device warm up
    with tf.device(device):
        _ = model(data)
        print('warm-up done.')

    # profile
    with tf.profiler.experimental.Profile(logdir=profiler_path):
        print('start profiling..')

        with tf.profiler.experimental.Trace(name='inference', step_num=0, _r=1):
            with tf.device(device):
                _ = model(data)

    print('end profiling.')


def get_threshold_at(efficiency: float, roc: dict, key='tpr') -> float:
    """Given a `roc` curve, it computes the threshold value to achieve the desired signal `efficiency`"""
    index = np.abs(roc[key] - efficiency).argmin()
    return roc['thresholds'][index]


def latents_as_scores(latents: dict, y: np.ndarray, masses: np.ndarray) -> dict:
    from ad.constants import BKG_INDEX
    scores = {}

    # loop over class labels
    for label in np.unique(y):
        mask = y == label
        name = utils.get_name(label).lower()
        score = {}

        # if is a signal
        if label != BKG_INDEX:
            # loop over mass labels, and select only the relevant latents
            for m, mass in utils.get_masses(label).items():
                score_m = {}
                mass_mask = masses == m

                for k, z in latents.items():
                    for i in range(z.shape[-1]):
                        score_m[f'{k}_{i + 1}'] = z[mask & mass_mask][:, i]

                score[mass] = score_m
        else:
            # QCD
            for k, z in latents.items():
                for i in range(z.shape[-1]):
                    score[f'{k}_{i + 1}'] = z[mask][:, i]

        scores[name] = score

    return scores


def average_predictions(model, x: Union[tuple, np.ndarray], y: np.ndarray, m: np.ndarray, cmap=None,
                        v_max: dict = None, x_index=0, **kwargs):
    from ad.constants import MASSES, LABELS, BKG_INDEX

    if cmap is None:
        cmap = ad.plot.DEFAULT_CMAP

    if isinstance(v_max, (int, float)):
        v_max = {i: float(v_max) for i, _ in enumerate(LABELS)}

    elif not isinstance(v_max, dict):
        v_max = {i: None for i, _ in enumerate(LABELS)}

    should_index = isinstance(x, (tuple, list))

    # y_pred = model.predict(x, **kwargs)
    mass_map = dict()

    # determine label-mass pairs
    for label in np.unique(y):

        if label in MASSES:
            # signal
            for m_ in utils.get_masses(label).keys():
                pair = (label, m_)
                mass_map[pair] = True
        else:
            # background
            pair = (label, None)
            mass_map[pair] = True

    for label, mass in mass_map.keys():
        if mass is None:
            # background
            print(utils.get_bkg_name())
            mass = BKG_INDEX
        else:
            # signal
            print(f'{utils.get_name(label)} ({utils.get_mass(label, mass)})')

        mask = (y == label) & (m == mass)
        print(f'#{mask.sum()}')

        x_mask = x[mask]
        y_pred = model.predict(x_mask, **kwargs)

        if not should_index:
            x_mu = np.mean(x_mask, axis=0)
        else:
            x_mu = np.mean(x[x_index][mask], axis=0)

        # y_mu = np.mean(y_pred[mask], axis=0)
        y_mu = np.mean(y_pred, axis=0)
        ad.plot.compare(x_mu, y_mu, cmap=cmap, v_max=v_max[int(label)])


def print_by_auc(curves: dict, return_results=False) -> list:
    results = []
    size_k = 0

    for k, curve in curves.items():
        auc = np.array([v['auc'] * 100.0 for v in curve.values()])
        avg = np.mean(auc).item()
        results.append((avg, k, auc))

        if len(k) > size_k:
            size_k = len(k)

    results = sorted(results, reverse=True)

    for avg, k, auc in results:
        print(f'[{k}]{"".ljust(size_k - len(k))} Avg: {round(avg, 2)}%, AUC: {np.round(auc, 2)}')

    if return_results:
        return results


def print_fpr_at_n_tpr(curves: dict, n=0.4, return_results=False, digits=4) -> list:
    results = []
    size_k = 0

    for k, curve in curves.items():
        fpr_at_n = np.array([utils.fpr_at_n_tpr(fpr=v['fpr'], tpr=v['tpr'], p=n) * 100.0 for v in curve.values()])
        avg = np.mean(fpr_at_n).item()
        results.append((avg, k, fpr_at_n))

        if len(k) > size_k:
            size_k = len(k)

    results = sorted(results, reverse=False)

    for avg, k, fpr_at_n in results:
        fpr_at_n = [round(v, digits) for v in fpr_at_n]
        print(f'[{k}]{"".ljust(size_k - len(k))} Avg: {round(avg, 2)}%, FPR@{int(n * 100)}: {fpr_at_n}')

    if return_results:
        return results


# ------------------------------------------------------------------------------------------------
# -- Anomaly Detection
# -------------------------------------------------------------------------------------------------

def cross_entropy_loss(true: tf.Tensor, pred: tf.Tensor) -> np.ndarray:
    loss = tf.keras.losses.binary_crossentropy(true, pred, axis=[])
    loss = tf.reduce_sum(loss, axis=[1, 2, 3])
    return loss.numpy()


def dice_loss(true: tf.Tensor, pred: tf.Tensor) -> np.ndarray:
    sum_p = tf.reduce_sum(tf.square(pred), axis=[1, 2, 3])
    sum_t = tf.reduce_sum(tf.square(true), axis=[1, 2, 3])

    union = sum_p + sum_t
    intersection = tf.reduce_sum(pred * true, axis=[1, 2, 3])

    loss = (union / (2 * intersection)).numpy()
    loss[np.isinf(loss) | np.isnan(loss)] = 0.0
    return loss


def mse_loss(true: tf.Tensor, pred: tf.Tensor) -> np.ndarray:
    loss = tf.square(true - pred)
    loss = tf.reduce_sum(loss, axis=[1, 2, 3])
    return loss.numpy()


def energy_loss(true: tf.Tensor, pred: tf.Tensor) -> np.ndarray:
    pred_energy = tf.reduce_sum(pred, axis=[1, 2, 3])
    true_energy = tf.reduce_sum(true, axis=[1, 2, 3])

    loss = []
    for p, t in zip(pred_energy, true_energy):
        loss.append(tf.keras.losses.huber(p[tf.newaxis], t[tf.newaxis]))

    return tf.stack(loss).numpy()


def diff_energy(true: tf.Tensor, pred: tf.Tensor) -> np.ndarray:
    pred_energy = tf.reduce_sum(pred, axis=[1, 2, 3])
    true_energy = tf.reduce_sum(true, axis=[1, 2, 3])

    return (true_energy - pred_energy).numpy()


def diff_abs_energy(true: tf.Tensor, pred: tf.Tensor) -> np.ndarray:
    scores = tf.math.abs(true - pred)
    scores = tf.reduce_sum(scores, axis=[1, 2, 3])
    return scores.numpy()


def diff_abs_mask(true: tf.Tensor, pred: tf.Tensor) -> np.ndarray:
    true_mask = tf.cast(true > 0.0, dtype=true.dtype)
    pred_mask = tf.cast(pred > 0.0, dtype=true.dtype)

    return diff_abs_energy(true=true_mask, pred=pred_mask)


def compute_scores(model, x: Union[tuple, np.ndarray], batch_size=128, x_index=0):
    """Scores computation for the AE model"""
    scores = dict(energy_diff=[], energy_pred=[], hits_diff=[], bce=[], dice=[], mse=[], total=[],
                  abs_diff=[], abs_mask=[])
    is_tuple = isinstance(x, (tuple, list))

    def index_fn(x):
        if is_tuple:
            return x[x_index]

        return x

    for batch in tf.data.Dataset.from_tensor_slices(x).batch(batch_size):
        y = model(batch)
        y_true = index_fn(batch)

        # scores
        pred_energy = tf.reduce_sum(y, axis=[1, 2, 3])
        energy_diff = tf.reduce_sum(y_true, axis=[1, 2, 3]) - pred_energy

        pred_hits = tf.reduce_sum(tf.cast(y > 0.0, dtype=y.dtype), axis=[1, 2, 3])
        hits_diff = tf.reduce_sum(tf.cast(y_true > 0.0, dtype=y.dtype), axis=[1, 2, 3]) - pred_hits

        abs_diff = diff_abs_energy(y_true, y)
        abs_mask = diff_abs_mask(y_true, y)

        bce = cross_entropy_loss(y, y_true)
        mse = mse_loss(y, y_true)
        dice = dice_loss(y, y_true)
        total = bce + dice

        # append scores
        for k, v in zip(scores.keys(), [energy_diff, pred_energy, hits_diff, bce, dice, mse, total,
                                        abs_diff, abs_mask]):
            scores[k].append(v)

    return {k: np.concatenate(v) for k, v in scores.items()}


def anomaly_scores(model, x: Union[tuple, np.ndarray], y: np.ndarray, m: np.ndarray, batch_size=128,
                   x_index=0, **kwargs) -> dict:
    from ad.constants import MASSES
    scores = {}
    is_tuple = isinstance(x, (tuple, list))

    def mask_fn(x, mask):
        if is_tuple:
            return tuple(x_[mask] for x_ in x)

        return x[mask]

    for label in np.unique(y):
        name = utils.get_name(label).lower()
        mask = y == label

        if label in MASSES:
            score = {}
            for m_, mass in utils.get_masses(label).items():
                score[mass] = compute_scores(model, x=mask_fn(x, mask=mask & (m == m_)),
                                             batch_size=batch_size, x_index=int(x_index))
        else:
            # QCD
            score = compute_scores(model, x=mask_fn(x, mask), batch_size=batch_size, x_index=int(x_index))

        scores[name] = score

    return scores


def compute_roc(bkg_scores: dict, signal_scores: dict, **kwargs) -> dict:
    from sklearn.metrics import roc_auc_score, roc_curve

    curves = dict()

    for k, bkg_score in bkg_scores.items():
        curves[k] = {}

        for h, score in signal_scores.items():
            sig_score = score[k]
            key = f'{utils.get_name_from(mass=h)} ({h})'

            # compute roc
            y_true = np.concatenate([
                np.zeros_like(bkg_score), np.ones_like(sig_score)])

            y_score = np.concatenate([bkg_score, sig_score])

            fpr, tpr, t = roc_curve(y_true, y_score, **kwargs)
            auc = roc_auc_score(y_true, y_score)

            curves[k][key] = dict(fpr=fpr, tpr=tpr, auc=auc, thresholds=t)

    return curves
