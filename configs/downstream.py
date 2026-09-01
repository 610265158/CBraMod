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
    'ema_decay',
    'num_workers',
    'optimizer',
    'label_smoothing',
    'binary_pos_weight',
    'dropout',
    'drop_path_rate',
    'early_stop',
    'frozen',
    'multi_lr',
    'use_pretrained_weights',
    'balanced_sampling',
    'balanced_sampling_power',
    'balanced_sampling_min_share',
    'mirror_augmentation',
    'mirror_prob',
    'time_roll_augmentation',
    'time_roll_prob',
    'time_roll_max_fraction',
    'amplitude_scale_augmentation',
    'amplitude_scale_prob',
    'amplitude_scale_min',
    'amplitude_scale_max',
    'amplitude_scale_distribution',
    'mixup_augmentation',
    'mixup_prob',
    'mixup_alpha',
    'amp',
    'amp_dtype',
    'mental_scale',
    'shu_clip_limit',
    'shu_scale',
    'shu_bandpass_low',
    'shu_bandpass_high',
    'shu_filter_order',
    'physio_lowpass_hz',
    'physio_filter_order',
    'mumtaz_lowpass_hz',
    'mumtaz_filter_order',
    'faced_input_norm',
    'faced_robust_clip',
    'test_each_epoch',
    'run_final_test',
    'selection_metric',
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
    'ema_decay': 0.0,
    'num_workers': 4,
    'optimizer': 'AdamW',
    'label_smoothing': 0.1,
    'binary_pos_weight': 1.0,
    'dropout': 0.1,
    'drop_path_rate': 0.0,
    'early_stop': 10,
    'frozen': False,
    'multi_lr': False,
    'use_pretrained_weights': True,
    'balanced_sampling': False,
    'balanced_sampling_power': 1.0,
    'balanced_sampling_min_share': 0.0,
    'mirror_augmentation': False,
    'mirror_prob': 0.5,
    'time_roll_augmentation': False,
    'time_roll_prob': 1.0,
    'time_roll_max_fraction': 0.5,
    'amplitude_scale_augmentation': False,
    'amplitude_scale_prob': 1.0,
    'amplitude_scale_min': 0.5,
    'amplitude_scale_max': 2.0,
    'amplitude_scale_distribution': 'log_uniform',
    'mixup_augmentation': False,
    'mixup_prob': 1.0,
    'mixup_alpha': 0.2,
    'amp': True,
    'amp_dtype': 'bfloat16',
    'mental_scale': 32.0,
    'shu_clip_limit': 512.0,
    'shu_scale': 64.0,
    'shu_bandpass_low': None,
    'shu_bandpass_high': None,
    'shu_filter_order': 4,
    'physio_lowpass_hz': None,
    'physio_filter_order': 4,
    'mumtaz_lowpass_hz': None,
    'mumtaz_filter_order': 4,
    'faced_input_norm': 'clip_scale',
    'faced_robust_clip': 8.0,
    'test_each_epoch': False,
    'run_final_test': True,
    'selection_metric': 'auto',
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


# BCIC2020-3, ISRUC, and MentalArithmetic are locked below. The remaining datasets
# still need either a five-seed confirmation under bottom/right padding or a
# finalized dataset-specific recipe before they can enter the formal table.
RERUN_REQUIRED_DATASETS = {
    'CHB-MIT': 'bottom_right_padding_5seed_pending',
    'TUAB': 'bottom_right_padding_5seed_pending',
    'TUEV': 'bottom_right_padding_5seed_pending',
    'FACED': 'recipe_search_pending',
    'SEED-V': 'recipe_search_pending',
    'PhysioNet-MI': 'dataset_recipe_5seed_pending',
    'SHU-MI': 'recipe_search_pending',
    'Mumtaz2016': 'dataset_split_and_recipe_pending',
}


def _training(**overrides):
    cfg = dict(DEFAULT_TRAINING)
    cfg.update(overrides)
    return cfg


def _vision(**overrides):
    cfg = deepcopy(DEFAULT_VISION)
    cfg.update(overrides)
    return cfg


# Locked five-seed recipes.  Keep these separate from the registry entries so
# later dataset-wide sweeps cannot silently replace the reported settings.
# Each result uses seeds 42--46, validation-selected checkpoints, population
# standard deviation, and one final test evaluation per seed.
FINALIZED_FIVE_SEED_RECIPES = {
    'BCIC2020-3': {
        'seeds': (42, 43, 44, 45, 46),
        'results': {
            'kappa': (0.58667, 0.00723),
            'f1': (0.66956, 0.00581),
        },
        'experiment_name': 'bcic2020_3_b0_p1_bottomrightpad_5seed_v1',
        'training': {
            'lr': 1e-3,
            'backbone_lr_scale': 1.0,
            'batch_size': 32,
            'num_workers': 4,
            'epochs': 30,
            'weight_decay': 5e-3,
            'min_lr': 1e-6,
            'warmup_epochs': 3,
            'warmup_start_factor': 0.1,
            'clip_value': 1.0,
            'ema_decay': 0.995,
            'optimizer': 'AdamW',
            'label_smoothing': 0.1,
            'dropout': 0.1,
            'drop_path_rate': 0.0,
            'early_stop': 30,
            'frozen': False,
            'multi_lr': False,
            'use_pretrained_weights': True,
            'balanced_sampling': False,
            # Imagined speech is hemispherically lateralized; left/right
            # channel mirroring is not assumed to preserve its labels.
            'mirror_augmentation': False,
            'time_roll_augmentation': True,
            'time_roll_prob': 1.0,
            'time_roll_max_fraction': 0.5,
            'amplitude_scale_augmentation': False,
            'mixup_augmentation': False,
            'amp': True,
            'amp_dtype': 'bfloat16',
            'test_each_epoch': False,
            'run_final_test': True,
            'selection_metric': 'kappa',
        },
        'vision': _vision(
            backbone_name='efficientnet_b0',
            adapter={'fold_factor': 1},
        ),
    },
    'ISRUC': {
        'seeds': (42, 43, 44, 45, 46),
        'results': {
            'ba': (0.80253, 0.00272),
            'kappa': (0.77045, 0.00316),
            'f1': (0.81970, 0.00287),
        },
        'experiment_name': 'isruc_p12_bottomrightpad_headstd002_5seed_v1',
        'training': {
            'lr': 1e-3,
            'backbone_lr_scale': 0.1,
            'batch_size': 16,
            'num_workers': 4,
            'epochs': 15,
            'weight_decay': 5e-3,
            'min_lr': 1e-6,
            'warmup_epochs': 3,
            'warmup_start_factor': 0.1,
            'clip_value': -1.0,
            'ema_decay': 0.995,
            'optimizer': 'AdamW',
            'label_smoothing': 0.1,
            'dropout': 0.1,
            'drop_path_rate': 0.0,
            'early_stop': 15,
            'frozen': False,
            'multi_lr': False,
            'use_pretrained_weights': True,
            'balanced_sampling': False,
            'mirror_augmentation': True,
            'mirror_prob': 0.5,
            'time_roll_augmentation': True,
            'time_roll_prob': 0.5,
            'time_roll_max_fraction': 0.25,
            'amplitude_scale_augmentation': False,
            'mixup_augmentation': False,
            'amp': True,
            'amp_dtype': 'bfloat16',
            'test_each_epoch': False,
            'run_final_test': True,
            'selection_metric': 'kappa',
        },
        'vision': _vision(
            backbone_name='efficientnet_b0',
            head_init_std=0.002,
            adapter={'fold_factor': 12},
        ),
    },
    'MentalArithmetic': {
        'seeds': (42, 43, 44, 45, 46),
        'results': {
            'pr_auc': (0.79029, 0.05364),
            'roc_auc': (0.87789, 0.02513),
        },
        'experiment_name': 'mentalarithmetic_p4_headstd002_ema995_5seed_v1',
        'training': {
            'lr': 1e-3,
            'backbone_lr_scale': 0.1,
            'batch_size': 64,
            'num_workers': 4,
            'epochs': 30,
            'weight_decay': 5e-4,
            'min_lr': 1e-6,
            'warmup_epochs': 3,
            'warmup_start_factor': 0.1,
            'clip_value': -1.0,
            'ema_decay': 0.995,
            'optimizer': 'AdamW',
            'label_smoothing': 0.1,
            'binary_pos_weight': 3.0,
            'dropout': 0.1,
            'drop_path_rate': 0.0,
            'early_stop': 30,
            'frozen': False,
            'multi_lr': False,
            'use_pretrained_weights': True,
            'balanced_sampling': False,
            'mirror_augmentation': True,
            'mirror_prob': 0.5,
            'time_roll_augmentation': True,
            'time_roll_prob': 0.5,
            'time_roll_max_fraction': 0.25,
            'amplitude_scale_augmentation': False,
            'mixup_augmentation': False,
            'amp': True,
            'amp_dtype': 'bfloat16',
            'mental_scale': 32.0,
            'test_each_epoch': False,
            'run_final_test': True,
            'selection_metric': 'pr_auc',
        },
        'vision': _vision(
            backbone_name='efficientnet_b0',
            squeeze_binary=True,
            head_init_std=0.002,
            adapter={'fold_factor': 4},
        ),
    },
}


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
        training={
            'lr': 1e-3,
            'batch_size': 32,
            'epochs': 10,
            'weight_decay': 5e-3,
            'selection_metric': 'pr_auc',
            'test_each_epoch': False,
            'run_final_test': True,
        },
        vision=_vision(squeeze_binary=True, init_head=False,
                       adapter={'fold_factor': 4}),
    ),
    'TUAB': _dataset(
        task='binary',
        classes=1,
        input_shape=(16, 2000),
        dataset_module='datasets.tuab_dataset',
        datasets_dir='../BigDownstream/TUAB',
        storage='pkl_split',
        split_dirs={'train': 'train', 'val': 'val', 'test': 'test'},
        training={
            'lr': 1e-3,
            'batch_size': 32,
            'epochs': 5,
            'weight_decay': 5e-4,
            'clip_value': 1,
            'selection_metric': 'pr_auc',
            'test_each_epoch': False,
            'run_final_test': True,
        },
        vision=_vision(squeeze_binary=True,
                       adapter={'fold_factor': 4}),
    ),
    'TUEV': _dataset(
        task='multiclass',
        classes=6,
        input_shape=(16, 1000),
        dataset_module='datasets.tuev_dataset',
        datasets_dir='../BigDownstream/TUEV_refine/processed',
        storage='pkl_split',
        split_dirs={'train': 'processed_train', 'val': 'processed_eval', 'test': 'processed_test'},
        training={
            'lr': 1e-3,
            'batch_size': 32,
            'epochs': 10,
            'weight_decay': 5e-3,
            'selection_metric': 'kappa',
            'test_each_epoch': False,
            'run_final_test': True,
        },
        vision=_vision(adapter={'fold_factor': 4}),
    ),
    'ISRUC': _dataset(
        task='multiclass',
        classes=5,
        input_shape=(20, 6, 6000),
        dataset_module='datasets.isruc_dataset',
        datasets_dir='../BigDownstream/ISRUC/precessed_filter_35',
        storage='isruc_npy',
        # Locked five-seed recipe (42--46). Each record contains 20 sleep
        # epochs, so a loader batch of 16 yields 320 independently classified
        # epochs. Padding is appended only at the bottom and right.
        training=FINALIZED_FIVE_SEED_RECIPES['ISRUC']['training'],
        vision=FINALIZED_FIVE_SEED_RECIPES['ISRUC']['vision'],
    ),
    'FACED': _dataset(
        task='multiclass',
        classes=9,
        input_shape=(32, 2000),
        dataset_module='datasets.faced_dataset',
        datasets_dir='../BigDownstream/faced/processed',
        storage='lmdb',
        training={
            'lr': 1e-3,
            'batch_size': 32,
            'epochs': 50,
            'weight_decay': 5e-3,
            'amp_dtype': 'bfloat16',
            'selection_metric': 'kappa',
            'test_each_epoch': False,
            'run_final_test': True,
        },
        vision=_vision(adapter={'fold_factor': 2}),
    ),
    'SEED-V': _dataset(
        task='multiclass',
        classes=5,
        input_shape=(62, 200),
        dataset_module='datasets.seedv_dataset',
        datasets_dir='../BigDownstream/SEED-V/processed',
        storage='lmdb',
        training={
            'lr': 5e-4,
            'batch_size': 32,
            'epochs': 50,
            'weight_decay': 5e-3,
            'amp_dtype': 'bfloat16',
            'selection_metric': 'kappa',
            'test_each_epoch': False,
            'run_final_test': True,
        },
        vision=_vision(adapter={'fold_factor': 1}),
    ),
    'PhysioNet-MI': _dataset(
        task='multiclass',
        classes=4,
        input_shape=(64, 800),
        dataset_module='datasets.physio_dataset',
        datasets_dir='../BigDownstream/eeg-motor-movementimagery-dataset-1.0.0',
        storage='lmdb',
        # Finalized P=1 stability recipe (seeds 3407/3408/3409). Motor-
        # imagery labels are side-specific, so left/right mirror augmentation
        # is not label preserving. Validation kappa selects one checkpoint,
        # which is evaluated exactly once on test for each reporting seed.
        training={
            'lr': 2e-3,
            'backbone_lr_scale': 0.1,
            'batch_size': 32,
            'epochs': 30,
            'weight_decay': 5e-3,
            'min_lr': 1e-6,
            'warmup_epochs': 3,
            'warmup_start_factor': 0.1,
            'clip_value': -1.0,
            'ema_decay': 0.995,
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
            'time_roll_augmentation': False,
            'amplitude_scale_augmentation': False,
            'amp': True,
            'amp_dtype': 'bfloat16',
            'test_each_epoch': False,
            'run_final_test': True,
            'selection_metric': 'kappa',
        },
        vision=_vision(adapter={'fold_factor': 1}),
    ),
    'SHU-MI': _dataset(
        task='binary',
        classes=1,
        input_shape=(32, 800),
        dataset_module='datasets.shu_dataset',
        datasets_dir='../BigDownstream/shu_datasets',
        storage='lmdb',
        # Finalized EfficientNet-B0 P=2 recipe after the B0/B5 three-seed
        # comparison.  Average pooling is retained; no runtime band-pass or
        # amplitude scaling is used.
        training={
            'lr': 1e-3,
            'backbone_lr_scale': 1.0,
            'batch_size': 32,
            'epochs': 20,
            'weight_decay': 5e-4,
            'min_lr': 1e-6,
            'warmup_epochs': 3,
            'warmup_start_factor': 0.1,
            'clip_value': -1.0,
            'ema_decay': 0.995,
            'optimizer': 'AdamW',
            'label_smoothing': 0.1,
            'binary_pos_weight': 1.0,
            'dropout': 0.1,
            'drop_path_rate': 0.0,
            'early_stop': 20,
            'frozen': False,
            'multi_lr': False,
            'use_pretrained_weights': True,
            'balanced_sampling': False,
            'mirror_augmentation': False,
            'time_roll_augmentation': True,
            'time_roll_prob': 0.5,
            'time_roll_max_fraction': 0.25,
            'amplitude_scale_augmentation': False,
            'amp': True,
            'amp_dtype': 'bfloat16',
            'test_each_epoch': False,
            'run_final_test': True,
            'selection_metric': 'pr_auc',
        },
        vision=_vision(squeeze_binary=True, adapter={'fold_factor': 2}),
    ),
    'BCIC2020-3': _dataset(
        task='multiclass',
        classes=5,
        input_shape=(64, 600),
        dataset_module='datasets.speech_dataset',
        datasets_dir='../BigDownstream/speech/processed',
        storage='lmdb',
        # Native height is already 64, so P=1 preserves the channel geometry.
        # Padding is appended only to the right: 64x600 -> 64x608.
        training=FINALIZED_FIVE_SEED_RECIPES['BCIC2020-3']['training'],
        vision=FINALIZED_FIVE_SEED_RECIPES['BCIC2020-3']['vision'],
    ),
    'Mumtaz2016': _dataset(
        task='binary',
        classes=1,
        input_shape=(19, 1000),
        dataset_module='datasets.mumtaz_dataset',
        datasets_dir='../BigDownstream/MDDPHCED/processed_lmdb_75hz',
        storage='lmdb',
        # Five-seed bottom/right-padding recipe (seeds 42--46).  The smaller
        # classifier initialization is matched to the stabilized
        # MentalArithmetic head while retaining the existing Mumtaz training
        # recipe.
        training={
            'lr': 1e-3,
            'backbone_lr_scale': 0.1,
            'batch_size': 32,
            'epochs': 30,
            'weight_decay': 5e-4,
            'min_lr': 1e-6,
            'warmup_epochs': 3,
            'warmup_start_factor': 0.1,
            'clip_value': -1.0,
            'ema_decay': 0.995,
            'optimizer': 'AdamW',
            'label_smoothing': 0.1,
            'binary_pos_weight': 1.0,
            'dropout': 0.1,
            'drop_path_rate': 0.0,
            'early_stop': 30,
            'frozen': False,
            'multi_lr': False,
            'use_pretrained_weights': True,
            'balanced_sampling': False,
            'mirror_augmentation': False,
            'time_roll_augmentation': True,
            'time_roll_prob': 1.0,
            'time_roll_max_fraction': 0.5,
            'amplitude_scale_augmentation': False,
            'amp': True,
            'amp_dtype': 'bfloat16',
            'test_each_epoch': False,
            'run_final_test': True,
            'selection_metric': 'pr_auc',
            # The LMDB is already band-pass filtered to 0.3--75 Hz during
            # preprocessing. Do not apply an additional runtime low-pass.
            'mumtaz_lowpass_hz': None,
            'mumtaz_filter_order': 4,
        },
        vision=_vision(squeeze_binary=True, head_init_std=0.002,
                       feature_aggregation='flatten',
                       adapter={'fold_factor': 4}),
    ),
    'MentalArithmetic': _dataset(
        task='binary',
        classes=1,
        input_shape=(20, 1000),
        dataset_module='datasets.stress_dataset',
        datasets_dir='../BigDownstream/mental-arithmetic/processed',
        storage='lmdb',
        # P=4 gives 80x250 and bottom/right padding gives 96x256.
        training=FINALIZED_FIVE_SEED_RECIPES['MentalArithmetic']['training'],
        vision=FINALIZED_FIVE_SEED_RECIPES['MentalArithmetic']['vision'],
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
