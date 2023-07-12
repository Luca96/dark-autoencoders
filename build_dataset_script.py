#!/usr/bin/python

import os
import h5py
import numpy as np
import tensorflow as tf
import multiprocessing as mp
import argparse
import gc

from sklearn.model_selection import train_test_split


def makedir(*args: str) -> str:
    """Creates a directory"""
    path = os.path.join(*args)
    os.makedirs(path, exist_ok=True)
    return path


def get_files(src: str, dst: str, should_split=False) -> list:
    files = os.listdir(src)

    if should_split:
        makedir(os.path.join(dst, 'train'))
        makedir(os.path.join(dst, 'valid'))
    else:
        makedir(dst)

    return files


def from_h5_to_npz(args: tuple):
        id, file_name, src, dst, split, normalize, should_sum = args
        dtype = np.float32

        should_split = split > 0.0
        path = os.path.join(src, file_name)
        print(f'[{id}] processing "{file_name}"..')

        # determine label
        if 'h125' in file_name:
            mass = 1
            label = 1

        elif 'h200' in file_name:
            mass = 2
            label = 1

        elif 'h300' in file_name:
            mass = 3
            label = 1

        elif 'h400' in file_name:
            mass = 2 + 2
            label = 1

        elif 'h700' in file_name:
            mass = 3 + 2
            label = 1

        elif 'h1000' in file_name:
            mass = 4 + 2
            label = 1

        elif 'svj' in file_name:
            label = 2

            # file_name has the format "svj_[mass]_[id]_[n].h5"
            part = file_name.split('_')[1]

            # TODO: should change mass label for SVJ; these are still unique if paired with class label
            mass = {2100: 5, 3100: 6, 4100: 7}[int(part)]
        else:
            mass = 0
            label = 0

        # each file contains N 286x360 images of the plane (eta, phi)
        with h5py.File(path, 'r') as file:
            # inner-tracker image (with pile-up correction)
            image_trk = np.array(file.get('ImageTrk_PUcorr'), dtype=dtype)

            # ECAL image
            image_ecal = np.array(file.get('ImageECAL'), dtype=dtype)

            # HCAL image
            image_hcal = np.array(file.get('ImageHCAL'), dtype=dtype)

            # number of tracks
            tracks = np.array(file.get('NTrk_PUcorr'), dtype=dtype)

            # stack the three images to form 3-channel images
            # shape: (N, 286, 360, 3)
            images = np.stack([image_trk, image_ecal, image_hcal], axis=-1).astype(dtype)

            # transpose to have (phi, eta) instead of (eta, phi)
            # shape: (N, 360, 286, 3)
            images = np.transpose(images, axes=[0, 2, 1, 3])
            labels = np.ones(len(images), dtype=dtype) * float(label)
            masses = np.ones(len(images), dtype=dtype) * float(mass)

            # pre-processing
            # down-sample (by 5x) and normalize images (to sum 1)
            with tf.device('cpu'):
                x = tf.nn.depthwise_conv2d(images, filter=tf.ones((5, 5, 3, 1)),
                                           strides=[1, 5, 5, 1], padding='SAME')
                if should_sum:
                    x = tf.reduce_sum(x, axis=-1, keepdims=True)

                if normalize:
                    x /= tf.reduce_sum(x, axis=[1, 2, 3], keepdims=True)

                images = np.array(x, dtype=dtype)

            # training-validation split
            if should_split:
                x_train, x_valid, \
                y_train, y_valid, \
                t_train, t_valid, \
                m_train, m_valid = train_test_split(images, labels, tracks, masses,
                                                    test_size=split, random_state=42)
                # save
                save_path_train = os.path.join(dst, 'train', file_name)
                save_path_valid = os.path.join(dst, 'valid', file_name)

                save_path_train, _ = os.path.splitext(save_path_train)  # remove .h5 extension
                save_path_valid, _ = os.path.splitext(save_path_valid)

                np.savez_compressed(save_path_train, images=x_train, labels=y_train,
                                    masses=m_train, n_tracks=t_train)

                np.savez_compressed(save_path_valid, images=x_valid, labels=y_valid,
                                    masses=m_valid, n_tracks=t_valid)

                print(f'  [{id}] saved at "{save_path_train}.npz"')
                print(f'  [{id}] saved at "{save_path_valid}.npz"')

                del x_train, x_valid, y_train, y_valid
                gc.collect()
            else:
                save_path = os.path.join(dst, file_name)
                save_path, _ = os.path.splitext(save_path)

                np.savez_compressed(save_path, images=images, labels=labels, masses=masses,
                                    n_tracks=tracks)

                print(f'  [{id}] saved at "{save_path}.npz"')

        # cleanup
        del file, image_trk, image_ecal, image_hcal, images, labels
        gc.collect()


if __name__ == '__main__':
    # NOTE: the QCD "valid" folder is manually copied in "test"

    # Call as follows:
    # QCD: src='../data/n_tracks/qcd', dst='../data/n_tracks-3c', split=0.4
    # SUEP: src='../data/n_tracks/suep', dst='../data/n_tracks-3c/test'
    # SVJ: src='../data/n_tracks/svj', dst='../data/n_tracks-3c/test'

    parser = argparse.ArgumentParser()

    # require arguments
    parser.add_argument('--src', required=True, type=str, help='source folder')
    parser.add_argument('--dst', required=True, type=str, help='destination folder')

    # optional arguments
    parser.add_argument('-s', '--split', default=0.0, type=float, help='% of train-valid split')
    parser.add_argument('--normalize', default=False, help='whether or not to normalize')
    parser.add_argument('--sum', default=False, help='whether or not to sum over channels')

    args = parser.parse_args()

    # processing
    files = get_files(src=args.src, dst=args.dst, should_split=args.split > 0.0)

    # prepare arguments for processes
    arguments = [(f'{i + 1}/{len(files)}', file, args.src, args.dst, args.split,
                  args.normalize, args.sum)
                 for i, file in enumerate(files)]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(from_h5_to_npz, arguments)

