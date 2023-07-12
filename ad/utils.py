import os
import gc
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py

from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime
from typing import List, Union

SEED = None


def get_name(label: int) -> str:
    """Returns the name of the given class label"""
    from ad.constants import LABELS
    return LABELS[int(label)]


def get_bkg_name() -> str:
    from ad.constants import LABELS, BKG_INDEX
    return LABELS[BKG_INDEX]


def get_label_idx(label: str) -> int:
    from ad.constants import LABELS
    assert label in LABELS

    for i, name in enumerate(LABELS):
        if label == name:
            return i


def get_label_from(mass_idx: int) -> str:
    from ad.constants import MASSES, BKG_INDEX
    label = BKG_INDEX

    for k, masses in MASSES.items():
        if mass_idx in masses:
            label = k
            break

    return label


def get_name_from(mass: int) -> str:
    from ad.constants import MASSES, LABELS, BKG_INDEX
    name = LABELS[BKG_INDEX]

    for k, masses in MASSES.items():
        if mass in masses.values():
            name = get_name(label=k)
            break

    return name


def get_masses(label: Union[int, str]) -> dict:
    from ad.constants import MASSES

    if isinstance(label, str):
        label = get_label_idx(label)
    else:
        label = int(label)

    # if label == 0:
    #     return []

    # from ad.constants import MASSES
    return MASSES.get(label, None)


def get_mass(label: Union[int, str], mass_idx: int) -> int:
    if isinstance(label, str):
        label = get_label_idx(label)

    if int(label) == 0:
        return 0

    return get_masses(label)[int(mass_idx)]


def set_random_seed(seed: int):
    """Sets the random seed for TensorFlow, numpy, python's random"""
    global SEED

    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        SEED = seed
        print(f'Random seed {SEED} set.')


def pad_images(images: np.ndarray, top=0, bottom=0, left=0, right=0, constant=0.0):
    """Pads a batch of images.
        - Based on: https://stackoverflow.com/a/51662925/21113996"""
    assert len(np.shape(images)) == 4, 'not a batch of images!'
    return np.pad(images, ((0, 0), (top, bottom), (left, right), (0, 0)),
                  mode='constant', constant_values=constant)


def tf_random_choice(size: int, seed=SEED):
    return tf.argmax(tf.random.uniform(shape=(size,), seed=seed, maxval=1.0))


def tf_random_choices(num_options: int, amount: int, seed=SEED) -> tf.Tensor:
    return tf.argmax(tf.random.uniform(shape=(amount, num_options), seed=seed, maxval=1.0), axis=-1)


def get_random_generator(seed=SEED) -> np.random.Generator:
    """Returns a numpy random generator instance"""
    if seed is not None:
        seed = int(seed)
        assert 0 <= seed < 2 ** 32

    return np.random.default_rng(np.random.MT19937(seed=seed))


def tf_global_norm(values: List[tf.Tensor]):
    """Computes the global l2-norm of a list of tensors"""
    # Euclidean norm of each item in the provided list
    local_norms = [tf.norm(v) for v in values]

    # now compute the global l2-norm
    return tf.sqrt(tf.reduce_sum([norm * norm for norm in local_norms]))


def free_mem():
    return gc.collect()


def normalize(x: np.ndarray, stats: tuple):
    x_min, x_max = stats
    return (x - x_max) / (x_max - x_min)


def fpr_at_n_tpr(fpr, tpr, p=0.4):
    """Return the FPR when TPR is at minimum p%.

    Source: https://github.com/tayden/ood-metrics/blob/main/ood_metrics/metrics.py#LL37C12-L37C12
    """
    if all(tpr < p):
        # No threshold allows TPR >= p
        return 0

    elif all(tpr >= p):
        # All thresholds allow TPR >= p, so find the lowest possible FPR
        idxs = [i for i, x in enumerate(tpr) if x >= p]
        return min(map(lambda idx: fpr[idx], idxs))

    # Linear interp between values to get FPR at TPR == p
    return np.interp(p, tpr, fpr)


def distance_corr(var_1: tf.Tensor, var_2: tf.Tensor, power=1):
    """Distance correlation (DisCo) from "ABCDisCo: Automating the ABCD Method with Machine Learning"
       Arguments:
           var_1: First variable to decorrelate (e.g. mass).
           var_2: Second variable to decorrelate (e.g. classifier output).
           power: Exponent used in calculating the distance correlation.

       va1_1, var_2 should all be 1D tf tensors with the same number of entries
    """
    x = tf.reshape(var_1, shape=(-1, 1))
    x = tf.tile(x, multiples=[1, tf.size(var_1)])
    x = tf.reshape(x, shape=[tf.size(var_1), tf.size(var_1)])

    y = tf.transpose(x)
    a_matrix = tf.math.abs(x - y)

    x = tf.reshape(var_2, shape=(-1, 1))
    x = tf.tile(x, multiples=[1, tf.size(var_2)])
    x = tf.reshape(x, shape=[tf.size(var_2), tf.size(var_2)])

    y = tf.transpose(x)
    b_matrix = tf.math.abs(x - y)

    a_mat_avg = tf.reduce_mean(a_matrix, axis=1)
    b_mat_avg = tf.reduce_mean(b_matrix, axis=1)

    minuend_1 = tf.tile(a_mat_avg, multiples=[tf.size(var_1)])
    minuend_1 = tf.reshape(minuend_1, shape=[tf.size(var_1), tf.size(var_1)])
    minuend_2 = tf.transpose(minuend_1)
    a_matrix = a_matrix - minuend_1 - minuend_2 + tf.reduce_mean(a_mat_avg)

    minuend_1 = tf.tile(b_mat_avg, multiples=[tf.size(var_2)])
    minuend_1 = tf.reshape(minuend_1, shape=[tf.size(var_2), tf.size(var_2)])
    minuend_2 = tf.transpose(minuend_1)
    b_matrix = b_matrix - minuend_1 - minuend_2 + tf.reduce_mean(b_mat_avg)

    ab_avg = tf.reduce_mean(a_matrix * b_matrix, axis=1)
    aa_avg = tf.reduce_mean(a_matrix * a_matrix, axis=1)
    bb_avg = tf.reduce_mean(b_matrix * b_matrix, axis=1)

    if power == 1:
        d_corr = tf.reduce_mean(ab_avg) / tf.sqrt(tf.reduce_mean(aa_avg) * tf.reduce_mean(bb_avg))
    elif power == 2:
        d_corr = (tf.reduce_mean(ab_avg))**2 / (tf.reduce_mean(aa_avg) * tf.reduce_mean(bb_avg))
    else:
        d_corr = (tf.reduce_mean(ab_avg) / tf.sqrt(tf.reduce_mean(aa_avg) * tf.reduce_mean(bb_avg)))**power

    return d_corr


def get_plot_axes(rows: int, cols: int, size=(12, 10), flatten_axes=False, **kwargs):
    rows = int(rows)
    cols = int(cols)

    assert rows >= 1
    assert cols >= 1

    fig, axes = plt.subplots(nrows=rows, ncols=cols, **kwargs)

    fig.set_figwidth(size[0] * cols)
    fig.set_figheight(size[1] * rows)

    if flatten_axes:
        return np.reshape(axes, newshape=[-1])

    return axes


def makedir(*args: str) -> str:
    """Creates a directory"""
    path = os.path.join(*args)
    os.makedirs(path, exist_ok=True)
    return path


def read_npz(folder: str, dtype=np.float16, limit: int = None, shuffle=False,
             verbose=True, keys='arr_0', keep: str = None):
    """Reads all the .npz files in the given `folder`, returning one big array"""
    should_skip = isinstance(keep, str)
    files = os.listdir(folder)

    if shuffle:
        random.shuffle(files)

    limit = np.inf if limit is None else int(limit)
    count = 0

    if isinstance(keys, (list, tuple)):
        assert len(keys) > 0

        if len(keys) == 1:
            dataset = []
        else:
            dataset = {k: [] for k in keys}
    else:
        assert isinstance(keys, str)
        keys = [keys]
        dataset = []

    for i, file_name in enumerate(files):
        if should_skip and (keep not in file_name):
            if verbose:
                print(f'skipped "{file_name}"')

            continue

        if verbose:
            print(f'[{i + 1}/{len(files)}] reading "{file_name}"..')

        path = os.path.join(folder, file_name)
        npz = np.load(path)

        if isinstance(dataset, dict):
            for k in keys:
                dataset[k].append(np.array(npz[k], dtype=dtype))

            count += len(dataset[keys[0]][-1])
        else:
            dataset.append(np.array(npz[keys[0]], dtype=dtype))
            count += len(dataset[-1])

        if count >= limit:
            if verbose:
                print(f'[break] limit of {limit} reached.')
            break

    # finally, stack each image over the batch dimension
    if isinstance(dataset, dict):
        return {k: np.concatenate(v, axis=0) for k, v in dataset.items()}

    return np.concatenate(dataset, axis=0)


def from_h5_to_npz(src: str, dst: str, dtype=np.float16):
    """Reads .h5 files and converts them into .npz"""
    makedir(dst)
    files = os.listdir(src)

    for i, file_name in enumerate(files):
        print(f'[{i + 1}/{len(files)}] reading "{file_name}"..')

        path = os.path.join(src, file_name)

        # each file contains N 286x360 images of the plane (eta, phi)
        with h5py.File(path, 'r') as file:
            # inner-tracker image
            image_trk = np.array(file.get('ImageTrk_PUcorr'), dtype=dtype)

            # ECAL image
            image_ecal = np.array(file.get('ImageECAL'), dtype=dtype)

            # HCAL image
            image_hcal = np.array(file.get('ImageHCAL'), dtype=dtype)

            # stack the three images to form 3-channel images
            # shape: (N, 286, 360, 3)
            images = np.stack([image_trk, image_ecal, image_hcal], axis=-1)

            # transpose to have (phi, eta) instead of (eta, phi)
            # shape: (N, 360, 286, 3)
            images = np.transpose(images, axes=[0, 2, 1, 3])

            # save
            save_path = os.path.join(dst, file_name)
            save_path, _ = os.path.splitext(save_path)  # remove .h5 extension

            np.savez_compressed(save_path, images)
            print(f'  -> saved at "{save_path}.npz"')

        # cleanup
        del file, image_trk, image_ecal, image_hcal, images
        free_mem()


# TODO: bug with negative values
def load_from_checkpoint(model: tf.keras.Model, path: str, mode: str, base_dir='weights', **kwargs):
    """Load the weights of a pre-built model"""
    path = os.path.join(base_dir, path)

    # list all files in directory
    files = os.listdir(path)

    # split into (path, ext) tuples
    files = [os.path.splitext(os.path.join(path, fname)) for fname in files]

    # keep only weights files
    files = filter(lambda x: 'data-' in x[1], files)

    # from tuples get only path; remove ext
    files = map(lambda x: x[0], files)

    # zip files with metric value
    files_and_metric = map(lambda x: (x, x.split('-')[-1]), files)

    # sort by metric value
    files = sorted(files_and_metric, key=lambda x: x[-1], reverse=mode.lower() == 'min')
    files = map(lambda x: x[0], files)
    files = list(files)

    # load the best weights
    print(f'Loaded from "{files[-1]}"')
    model.load_weights(files[-1], **kwargs)


def delete_checkpoints(path: str, base='weights', mode='max'):
    """Keeps only the best checkpoint while deleting the others"""
    path = os.path.join(base, path)

    # list all files in directory
    files = os.listdir(path)

    # split into (path, ext) tuples
    files = [os.path.splitext(os.path.join(path, fname)) for fname in files]

    # keep only weights files
    files = filter(lambda x: 'data-' in x[1], files)

    # from tuples get only path; remove ext
    files = map(lambda x: x[0], files)

    # zip files with metric value
    files_and_metric = map(lambda x: (x, float(x.split('-')[-1])), files)

    # sort by metric value
    files = sorted(files_and_metric, key=lambda x: x[-1], reverse=mode.lower() == 'min')
    files = map(lambda x: x[0], files)
    files = list(files)

    # load the best weights
    print(f'Keep "{files[-1]}"')

    for f in files[:-1]:
        os.remove(f + '.index')
        os.remove(f + '.data-00000-of-00001')
        print(f'Deleted {f}.')


def get_checkpoint(path: str, monitor: str, mode: str, best_only=True):
    path = os.path.join('weights', path, 'weights-{epoch:02d}-' + f'\u007b{monitor}:.3f\u007d')

    return ModelCheckpoint(path,
                           save_weights_only=True, monitor=monitor,
                           mode=mode, save_best_only=bool(best_only))


def get_tensorboard(folder: str, **kwargs):
    logdir = f"logs/{folder}/" + actual_datetime()
    return tf.keras.callbacks.TensorBoard(log_dir=logdir,
                                          write_graph=False, **kwargs)


def actual_datetime() -> str:
    """Returns the current data timestamp, formatted as follows: YearMonthDay-HourMinuteSecond"""
    return datetime.now().strftime("%Y%m%d-%H%M%S")
