from copy import deepcopy


TRAINING_KEYS = (
    'lr',
    'backbone_lr_scale',
    'batch_size',
    'epochs',
    'weight_decay',
    'min_lr',
    'warmup_epochs',
    'warmup_start_factor',
    'clip_value',
    'num_workers',
    'optimizer',
    'label_smoothing',
    'dropout',
    'drop_path_rate',
    'early_stop',
    'frozen',
    'multi_lr',
    'use_pretrained_weights',
    'balanced_sampling',
    'mirror_augmentation',
    'mirror_prob',
    'time_roll_augmentation',
    'time_roll_prob',
    'time_roll_max_fraction',
    'amplitude_scale_augmentation',
    'amplitude_scale_prob',
    'amplitude_scale_min',
    'amplitude_scale_max',
    'amp',
    'amp_dtype',
    'shu_clip_limit',
    'shu_scale',
    'faced_input_norm',
    'faced_robust_clip',
)

DEFAULT_TRAINING = {
    'lr': 0.0005,
    'backbone_lr_scale': 0.1,
    'batch_size': 32,
    'epochs': 50,
    'weight_decay': 5e-2,
    'min_lr': 1e-6,
    'warmup_epochs': 0,
    'warmup_start_factor': 0.1,
    'clip_value': -1,
    'num_workers': 4,
    'optimizer': 'AdamW',
    'label_smoothing': 0.1,
    'dropout': 0.1,
    'drop_path_rate': 0.0,
    'early_stop': 10,
    'frozen': False,
    'multi_lr': False,
    'use_pretrained_weights': True,
    'balanced_sampling': False,
    'mirror_augmentation': False,
    'mirror_prob': 0.5,
    'time_roll_augmentation': False,
    'time_roll_prob': 1.0,
    'time_roll_max_fraction': 0.5,
    'amplitude_scale_augmentation': False,
    'amplitude_scale_prob': 1.0,
    'amplitude_scale_min': 0.5,
    'amplitude_scale_max': 2.0,
    'amp': True,
    'amp_dtype': 'float16',
    'shu_clip_limit': 512.0,
    'shu_scale': 64.0,
    'faced_input_norm': 'clip_scale',
    'faced_robust_clip': 8.0,
}


MODEL_TRAINING_OVERRIDES = {
    'vision': {
        # Keep efficientnet_b0 on the dataset/default training schedule.
        'efficientnet_b0': {},
        # convformer_s18 diverged with lr=5e-4 on CHB-MIT; 1e-4 is stable.
        'convformer_s18': {
            'lr': 0.0001,
        },
    },
    'eegnet': {
        '_default': {
            'lr': 0.001,
            'weight_decay': 0.0,
            'optimizer': 'Adam',
            'label_smoothing': 0.0,
            'dropout': 0.5,
            'multi_lr': False,
            'use_pretrained_weights': False,
        },
    },
}

DEFAULT_VISION = {
    'backbone_name': 'efficientnet_b0',
    'adapter': {
        'fold_factor': 2,
    },
}


def _training(**overrides):
    cfg = dict(DEFAULT_TRAINING)
    cfg.update(overrides)
    return cfg


def _vision(**overrides):
    cfg = deepcopy(DEFAULT_VISION)
    cfg.update(overrides)
    return cfg


def _dataset(
        task,
        classes,
        input_shape,
        dataset_module,
        datasets_dir,
        storage,
        split_dirs=None,
        training=None,
        vision=None,
):
    training = _training(**(training or {}))
    cfg = {
        'task': task,
        'classes': classes,
        'input_shape': input_shape,
        'dataset_module': dataset_module,
        'datasets_dir': datasets_dir,
        'storage': storage,
        'training': training,
        'vision': vision or _vision(),
    }
    if split_dirs is not None:
        cfg['split_dirs'] = split_dirs
    cfg.update(training)
    return cfg


DOWNSTREAM_11_CONFIGS = {
    'CHB-MIT': _dataset(
        task='binary',
        classes=1,
        input_shape=(16, 2000),
        dataset_module='datasets.chb_dataset',
        datasets_dir='../BigDownstream/chb-mit/processed_seg',
        storage='pkl_split',
        split_dirs={'train': 'train', 'val': 'val', 'test': 'test'},
        training={'epochs': 10},
        vision=_vision(squeeze_binary=True, init_head=False,
                       adapter={'fold_factor': 8}),
    ),
    'TUAB': _dataset(
        task='binary',
        classes=1,
        input_shape=(16, 2000),
        dataset_module='datasets.tuab_dataset',
        datasets_dir='../BigDownstream/TUAB',
        storage='pkl_split',
        split_dirs={'train': 'train', 'val': 'val', 'test': 'test'},
        training={'clip_value': 1, 'epochs': 5},
        vision=_vision(squeeze_binary=True,
                       adapter={'fold_factor': 8}),
    ),
    'TUEV': _dataset(
        task='multiclass',
        classes=6,
        input_shape=(16, 1000),
        dataset_module='datasets.tuev_dataset',
        datasets_dir='../BigDownstream/TUEV_refine/processed',
        storage='pkl_split',
        split_dirs={'train': 'processed_train', 'val': 'processed_eval', 'test': 'processed_test'},
    ),
    'ISRUC': _dataset(
        task='multiclass',
        classes=5,
        input_shape=(20, 6, 6000),
        dataset_module='datasets.isruc_dataset',
        datasets_dir='../BigDownstream/ISRUC/precessed_filter_35',
        storage='isruc_npy',
        training={'batch_size': 16},
        vision=_vision(adapter={'fold_factor': 4}),
    ),
    'FACED': _dataset(
        task='multiclass',
        classes=9,
        input_shape=(32, 2000),
        dataset_module='datasets.faced_dataset',
        datasets_dir='../BigDownstream/faced/processed',
        storage='lmdb',
    ),
    'SEED-V': _dataset(
        task='multiclass',
        classes=5,
        input_shape=(62, 200),
        dataset_module='datasets.seedv_dataset',
        datasets_dir='../BigDownstream/SEED-V/processed',
        storage='lmdb',
        vision=_vision(adapter={'fold_factor': 2}),
    ),
    'PhysioNet-MI': _dataset(
        task='multiclass',
        classes=4,
        input_shape=(64, 800),
        dataset_module='datasets.physio_dataset',
        datasets_dir='../BigDownstream/eeg-motor-movementimagery-dataset-1.0.0',
        storage='lmdb',
        training={'epochs': 20},
        vision=_vision(adapter={'fold_factor': 2}),
    ),
    'SHU-MI': _dataset(
        task='binary',
        classes=1,
        input_shape=(32, 800),
        dataset_module='datasets.shu_dataset',
        datasets_dir='../BigDownstream/shu_datasets',
        storage='lmdb',
        training={'epochs': 20},
        vision=_vision(squeeze_binary=True, adapter={'fold_factor': 2}),
    ),
    'BCIC2020-3': _dataset(
        task='multiclass',
        classes=5,
        input_shape=(64, 600),
        dataset_module='datasets.speech_dataset',
        datasets_dir='../BigDownstream/speech/processed',
        storage='lmdb',
        vision=_vision(adapter={'fold_factor': 2}),
    ),
    'Mumtaz2016': _dataset(
        task='binary',
        classes=1,
        input_shape=(19, 1000),
        dataset_module='datasets.mumtaz_dataset',
        datasets_dir='../BigDownstream/MDDPHCED/processed_lmdb_75hz',
        storage='lmdb',
        vision=_vision(squeeze_binary=True, adapter={'fold_factor': 4}),
    ),
    'MentalArithmetic': _dataset(
        task='binary',
        classes=1,
        input_shape=(20, 1000),
        dataset_module='datasets.stress_dataset',
        datasets_dir='../BigDownstream/mental-arithmetic/processed',
        storage='lmdb',
        vision=_vision(squeeze_binary=True, adapter={'fold_factor': 4}),
    ),
}


def get_dataset_config(name):
    if name not in DOWNSTREAM_11_CONFIGS:
        raise KeyError('Unknown downstream dataset: {}'.format(name))
    return deepcopy(DOWNSTREAM_11_CONFIGS[name])


def training_config_for(name, model_arch='vision', backbone_name=None):
    cfg = get_dataset_config(name)
    training = dict(cfg['training'])
    training.update(model_training_overrides(cfg, model_arch, backbone_name))
    return training


def model_training_overrides(cfg, model_arch='vision', backbone_name=None):
    arch_overrides = MODEL_TRAINING_OVERRIDES.get(model_arch, {})
    overrides = dict(arch_overrides.get('_default', {}))
    if model_arch == 'vision':
        if backbone_name is None:
            backbone_name = cfg.get('vision', {}).get('backbone_name')
        overrides.update(arch_overrides.get(backbone_name, {}))
    return overrides


def dataset_registry():
    return {
        name: {
            'dataset_module': cfg['dataset_module'],
            'task': cfg['task'],
        }
        for name, cfg in DOWNSTREAM_11_CONFIGS.items()
    }
