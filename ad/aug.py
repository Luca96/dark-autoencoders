"""Data augmentations"""

import tensorflow as tf

from ad import utils


def flip_eta(x):
    return tf.cast(tf.image.flip_left_right(x), dtype=tf.float32)


def tf_rotate_phi_down(x, amount: int):
    h = tf.shape(x)[0]

    # slice the input
    bottom = x[h - amount:, :, :]
    top = x[:h - amount, :, :]

    return tf.cast(tf.concat([bottom, top], axis=0), dtype=tf.float32)


def tf_rotate_phi_up(x, amount: int):
    # slice the input
    top = x[:amount, :, :]
    bottom = x[amount:, :, :]

    return tf.cast(tf.concat([bottom, top], axis=0), dtype=tf.float32)


def tf_augment(x, delta=8, size=7, seed=utils.SEED):
    # we have 6 possible choices
    choice = utils.tf_random_choice(size=6, seed=seed)
    amount = delta * (1 + utils.tf_random_choice(size=int(size), seed=seed))  # up-to amount of 56
    amount = tf.cast(amount, dtype=tf.int32)

    if choice == 1:
        # flip in eta
        return flip_eta(x)

    if choice == 2:
        # downward rotation in phi
        return tf_rotate_phi_down(x, amount=amount)

    if choice == 3:
        # upward rotation in phi
        return tf_rotate_phi_up(x, amount=amount)

    if choice == 4:
        # flip + down rotation
        return tf_rotate_phi_down(flip_eta(x), amount=amount)

    if choice == 5:
        # flip + up rotation
        return tf_rotate_phi_up(flip_eta(x), amount=amount)

    # no augmentation
    return tf.cast(x, dtype=tf.float32)


def tf_cutout(x, amount=(1, 2)):
    h, w, c = x.shape

    size = amount[0] + utils.tf_random_choice(size=amount[1])
    size = tf.cast(size, dtype=tf.int32)

    # create a mask that has size x size zones
    mask = tf.range(size * size, dtype=x.dtype)
    mask = tf.reshape(mask, shape=(size, size, 1))
    # mask = tf.stack([mask] * c)

    # one zone is randomly zeroed
    position = utils.tf_random_choice(size * size)
    position = tf.cast(position, dtype=x.dtype)

    mask = tf.cast(mask != position, dtype=x.dtype)
    mask = tf.image.resize([mask], size=(h, w),
                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)[0]
    return x * mask


def tf_chance(seed=None):
    """Use to get a single random number between 0 and 1"""
    return tf.random.uniform(shape=(1,), minval=0.0, maxval=1.0, seed=seed)
