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
