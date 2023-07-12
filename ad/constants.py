
import numpy as np


# keys of each `.h5` file
KEYS = ['ImageECAL', 'ImageECAL_noPU', 'ImageHCAL',
        'ImageHCAL_noPU', 'ImageTrk', 'ImageTrk_PUcorr',
        'ImageTrk_noPU']

STR_TO_CLASS = {'qcd': 0, 'h125': 1, 'h400': 2, 'h700': 3, 'h1000': 4}

# maximum value (of train-set) of each image channel
MAX_VAL = np.array([6770.0, 2192.0, 115.56])

# minimum value (excluding non-zero pixels) computed on normalized images
MIN_VAL = np.array([1.043e-5, 0.0002031, 0.000516])


MIN_ENERGY = np.array([0.08, 0.4, 0.05], dtype=np.float32)
MAX_CLIPPED_ENERGY = np.array([250.0, 100.0, 100.0], dtype=np.float32)

BKG_INDEX = 0
LABELS = ['QCD', 'SUEP', 'SVJ']
MASSES = {1: {1: 125, 2: 200, 3: 300, 4: 400, 5: 700, 6: 1000},  # SUEP (GeV)
          2: {5: 2.1, 6: 3.1, 7: 4.1}}  # SVJ (TeV)


def set_labels(labels: list, bkg_index: int):
    global LABELS, BKG_INDEX
    assert len(labels) > 0

    LABELS = labels
    BKG_INDEX = int(bkg_index)


def set_masses(masses: dict):
    global MASSES
    MASSES = masses


def clip(x):
    return np.clip(x, 0.0, 2000.0) / 2000.0


CLIP_MAX_3C = [1500.0, 750.0, 1000.0]
CLIP_MIN_3C = [0.07080078, 0.4453125, 0.8944702]


def clip_3c(x):
    return np.clip(x, 0.0, CLIP_MAX_3C) / CLIP_MAX_3C
