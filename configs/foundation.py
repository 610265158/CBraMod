"""Input specifications for the foundation-model matched-pipeline rerun.

Both foundation models consume the same stored arrays as the vision pipeline,
but each foundation model expects its own preprocessing convention.  For
foundation runs the dataset loaders are re-configured (through
``datasets.shape_utils.configure_eeg_normalization``) so every batch is
normalised exactly as the released preprocessing does:

* CBraMod divides raw microvolt samples by 100 for every downstream dataset.
* REVE uses the per-dataset ``scale_factor`` of its released task configs
  (1000 for FACED, 100 for TUAB/MUMTAZ, 10 for HMC).  TUAB is fed through the
  16-channel bipolar representation the local benchmark inherits from CBraMod;
  the released 21-channel memmap applies an equivalent microvolt / 100 scale.

Neither released pipeline clips the samples, so clipping is disabled for
foundation runs (``limit=inf``); the vision pipeline keeps its default
clip +/-1024 and divide-by-32.

Electrode names mirror the released REVE task configs.  Bipolar names such as
"FP1-F7" are averaged by the position-bank loader, following the released
``position_utils`` behaviour.

Eleven of the twelve datasets now carry specs.  The first five (FACED, TUAB,
HMC, Mumtaz2016, PhysioNet-MI) use each model's released convention exactly as
before.  The extension added on 2026-09-19 covers the remaining seven tasks:

* CBraMod keeps microvolt / 100 everywhere (the released convention is
  dataset-independent).
* REVE follows the released task configs where they exist: TUEV uses the same
  volt-scale memmap factor as TUAB (``x1e4``, equivalent to microvolt / 100 on
  the stored arrays), ISRUC uses the released loader constant (``/ 10``),
  Speech (BCIC2020-3) and Mental Arithmetic use their released ``scale_factor``
  of 1000.
* CHB-MIT, SEED-V and SHU-MI are NOT part of the released REVE benchmark, so
  their REVE specs are fallbacks: microvolt / 100 (the CBraMod-equivalent
  scale) with the pooled-token readout, except SEED-V where that readout stays
  at chance and the non-pooling readout is used.  They are reference-only runs
  and must be reported as such.  The ``dropout`` field of every REVE spec is informational:
  ``finetune_main`` resolves the classifier dropout from the per-dataset
  training block in ``configs/downstream.py`` before the model is built.
* The REVE position bank has no cerebellar electrodes; SEED-V's CB1/CB2 are
  mapped to the inferior occipital positions OI1h/OI2h, the closest available
  coordinates.
"""

from copy import deepcopy


CBRA_MOD_CHECKPOINT = 'pretrained_weights/pretrained_weights.pth'

HMC_ELECTRODES = ['F4', 'C4', 'O2', 'C3']

MUMTAZ_ELECTRODES = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                     'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ']

FACED_ELECTRODES = ['FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6',
                    'CZ', 'C3', 'C4', 'T3', 'T4', 'CP1', 'CP2', 'CP5', 'CP6', 'PZ', 'P3',
                    'P4', 'T5', 'T6', 'PO3', 'PO4', 'OZ', 'O1', 'O2', 'A2', 'A1']

TUAB_ELECTRODES = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
                   'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2']

PHYSIO_ELECTRODES = ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'CZ', 'C2',
                     'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'FP1', 'Fpz', 'FP2',
                     'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6',
                     'F8', 'FT7', 'FT8', 'T3', 'T4', 'T9', 'T10', 'TP7', 'TP8', 'T5', 'P5', 'P3', 'P1',
                     'PZ', 'P2', 'P4', 'P6', 'T6', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'OZ', 'O2', 'Iz']

# Released REVE configs use the same 16 bipolar pairs for TUEV and TUAB; the
# local TUEV arrays follow that order (preprocessing/preprocessing_tuev.py).
TUEV_ELECTRODES = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
                   'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2']

# CHB-MIT is not part of the released REVE benchmark; the local arrays use the
# T7/P7 bipolar naming of preprocessing/CHB-MIT/process2.py.
CHB_MIT_ELECTRODES = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
                      'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2']

# SEED-V is not part of the released REVE benchmark; the local 62-channel order
# comes from datasets/channel_mirror.py.  The position bank has no cerebellar
# entries, so CB1/CB2 fall back to the inferior occipital OI1h/OI2h.
SEEDV_ELECTRODES = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4',
                    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
                    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
                    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
                    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
                    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
                    'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
                    'OI1h', 'O1', 'OZ', 'O2', 'OI2h']

# SHU-MI is not part of the released REVE benchmark; the local 32-channel order
# comes from datasets/channel_mirror.py.
SHU_MI_ELECTRODES = ['FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8',
                     'FC1', 'FC2', 'FC5', 'FC6',
                     'CZ', 'C3', 'C4', 'T3', 'T4', 'A1', 'A2',
                     'CP1', 'CP2', 'CP5', 'CP6',
                     'PZ', 'P3', 'P4', 'T5', 'T6',
                     'PO3', 'PO4', 'OZ', 'O1', 'O2']

# Released REVE "Speech" (BCIC2020) electrode order.
BCIC2020_3_ELECTRODES = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
                         'FC5', 'FC1', 'FC2', 'FC6',
                         'T3', 'C3', 'CZ', 'C4', 'T4',
                         'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10',
                         'T5', 'P3', 'PZ', 'P4', 'T6',
                         'PO9', 'O1', 'OZ', 'O2', 'PO10',
                         'AF7', 'AF3', 'AF4', 'AF8',
                         'F5', 'F1', 'F2', 'F6',
                         'FT9', 'FT7', 'FC3', 'FC4', 'FT8', 'FT10',
                         'C5', 'C1', 'C2', 'C6',
                         'TP7', 'CP3', 'CPz', 'CP4', 'TP8',
                         'P5', 'P1', 'P2', 'P6',
                         'PO7', 'PO3', 'POz', 'PO4', 'PO8']

# Released REVE "Mental Arithmetic" (stress) electrode order.
MENTAL_ARITHMETIC_ELECTRODES = ['FP1', 'FP2', 'F3', 'F4', 'F7', 'F8', 'T3', 'T4', 'C3', 'C4',
                                'T5', 'T6', 'P3', 'P4', 'O1', 'O2', 'FZ', 'CZ', 'PZ', 'A2']

# Released REVE ISRUC electrode order; the local arrays store the same six
# derivations (F3-A2, C3-A2, O1-A2, F4-A1, C4-A1, O2-A1).
ISRUC_ELECTRODES = ['F3', 'C3', 'O1', 'F4', 'C4', 'O2']


FOUNDATION_SPECS = {
    'FACED': {
        'cbramod': {'input_scale': 100.0},
        'reve': {
            'input_scale': 1000.0,
            'electrodes': FACED_ELECTRODES,
            'pooling': 'no',
            'dropout': 0.1,
        },
    },
    'TUAB': {
        'cbramod': {'input_scale': 100.0},
        'reve': {
            'input_scale': 100.0,
            'electrodes': TUAB_ELECTRODES,
            'pooling': 'no',
            'dropout': 0.3,
        },
    },
    'HMC': {
        'cbramod': {'input_scale': 100.0},
        'reve': {
            'input_scale': 10.0,
            'electrodes': HMC_ELECTRODES,
            'pooling': 'last',
            'dropout': 0.3,
        },
    },
    'Mumtaz2016': {
        'cbramod': {'input_scale': 100.0},
        'reve': {
            'input_scale': 100.0,
            'electrodes': MUMTAZ_ELECTRODES,
            'pooling': 'last',
            'dropout': 0.5,
        },
    },
    'PhysioNet-MI': {
        'cbramod': {'input_scale': 100.0},
        'reve': {
            'input_scale': 100.0,
            'electrodes': PHYSIO_ELECTRODES,
            'pooling': 'no',
            'dropout': 0.5,
        },
    },
    'CHB-MIT': {
        'cbramod': {'input_scale': 100.0},
        'reve': {
            'input_scale': 100.0,
            'electrodes': CHB_MIT_ELECTRODES,
            'pooling': 'last',
            'dropout': 0.5,
        },
    },
    'TUEV': {
        'cbramod': {'input_scale': 100.0},
        'reve': {
            'input_scale': 100.0,
            'electrodes': TUEV_ELECTRODES,
            'pooling': 'no',
            'dropout': 0.15,
        },
    },
    'SEED-V': {
        'cbramod': {'input_scale': 100.0},
        'reve': {
            # The pooled-token readout stays at chance on this fallback task
            # (five-seed kappa 0.078); the non-pooling readout is used instead.
            'input_scale': 100.0,
            'electrodes': SEEDV_ELECTRODES,
            'pooling': 'no',
            'dropout': 0.5,
        },
    },
    'SHU-MI': {
        'cbramod': {'input_scale': 100.0},
        'reve': {
            'input_scale': 100.0,
            'electrodes': SHU_MI_ELECTRODES,
            'pooling': 'last',
            'dropout': 0.5,
        },
    },
    'BCIC2020-3': {
        'cbramod': {'input_scale': 100.0},
        'reve': {
            # The released Speech config selects the pooled-token readout, but on
            # these arrays it stays at chance (five-seed kappa 0.107 under the
            # campaign recipe); the non-pooling readout converges above the
            # published value, so it is used here (see the dataset record notes).
            'input_scale': 1000.0,
            'electrodes': BCIC2020_3_ELECTRODES,
            'pooling': 'no',
            'dropout': 0.5,
        },
    },
    'ISRUC': {
        'cbramod': {'input_scale': 100.0},
        'reve': {
            'input_scale': 10.0,
            'electrodes': ISRUC_ELECTRODES,
            'pooling': 'last',
            'dropout': 0.3,
        },
    },
    'MentalArithmetic': {
        'cbramod': {'input_scale': 100.0},
        'reve': {
            'input_scale': 1000.0,
            'electrodes': MENTAL_ARITHMETIC_ELECTRODES,
            'pooling': 'last',
            'dropout': 0.5,
        },
    },
}


def foundation_input_scale(model_arch, dataset_name):
    if dataset_name not in FOUNDATION_SPECS:
        raise KeyError(
            'No foundation input spec for dataset {}; add one to configs/foundation.py'.format(dataset_name)
        )
    return FOUNDATION_SPECS[dataset_name][model_arch]['input_scale']


def foundation_spec(dataset_name):
    if dataset_name not in FOUNDATION_SPECS:
        raise KeyError(
            'No foundation input spec for dataset {}; add one to configs/foundation.py'.format(dataset_name)
        )
    return deepcopy(FOUNDATION_SPECS[dataset_name])


def foundation_channels_and_time(input_shape, dataset_name):
    """Return (channels, time_steps) for a [C,T] or ISRUC [S,C,T] stored shape."""
    shape = tuple(input_shape)
    if len(shape) == 3:
        return shape[1], shape[2]
    if len(shape) == 2:
        return shape[0], shape[1]
    raise ValueError(
        'Unsupported input shape {} for dataset {}'.format(shape, dataset_name)
    )
