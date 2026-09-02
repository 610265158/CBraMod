#!/usr/bin/env python
import argparse
import importlib.util
import os
import pickle
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.downstream import DOWNSTREAM_11_CONFIGS as EXPERIMENTS
from configs.downstream import RERUN_REQUIRED_DATASETS
from configs.downstream import TRAINING_KEYS
from configs.downstream import training_config_for
from configs.backbones import backbone_name_for, load_backbone_config
from configs.experiment import load_experiment_config


def main():
    parser = argparse.ArgumentParser(description='Run the configured downstream experiments.')
    parser.add_argument('--dataset', action='append', choices=sorted(EXPERIMENTS.keys()),
                        help='dataset to run; can be specified multiple times')
    parser.add_argument('--config', type=str, default=None,
                        help='complete experiment YAML (dataset, model, recipe, protocol, output)')
    parser.add_argument('--all', action='store_true', help='run all configured datasets sequentially')
    parser.add_argument('--list', action='store_true', help='print configured datasets')
    parser.add_argument('--check_only', action='store_true', help='check paths and dependencies without training')
    parser.add_argument('--dry_run', action='store_true',
                        help='pass --dry_run to finetune_main.py for one-batch forward validation')
    parser.add_argument('--python', default=sys.executable, help='python executable used to call finetune_main.py')
    parser.add_argument('--seed', type=int, default=None,
                        help='run only this seed; overrides protocol.seeds for --config')
    parser.add_argument('--model_root', default='./experiments/checkpoints',
                        help='directory that will contain per-dataset weights')
    parser.add_argument('--log_root', default='./experiments/logs',
                        help='directory that will contain per-run logs')
    parser.add_argument('--no_log_file', action='store_true',
                        help='stream output only; do not write an experiment log file')
    parser.add_argument('--model_arch', choices=['vision', 'eegnet'], default='vision',
                        help='downstream model architecture')
    parser.add_argument('--backbone_name', type=str, default=None,
                        help='override timm backbone name for --model_arch vision')
    parser.add_argument('--backbone_config', type=str, default=None,
                        help='backbone profile name or YAML path (e.g. convnextv2_tiny)')
    parser.add_argument('--vision_fold_factor', type=int, default=None,
                        help='override phase-interleaved temporal fold factor P (minimum: 1)')
    parser.add_argument('--vision_channel_repeat', type=int, default=None,
                        help='repeat each EEG channel before phase folding')
    parser.add_argument('--vision_height_stride', type=int, default=None,
                        choices=[1, 2, 4, 8, 16, 32],
                        help='target CNN output stride along EEG-channel height')
    parser.add_argument('--vision_no_pad', action='store_true',
                        help='disable zero-padding after EEG phase folding')
    parser.add_argument('--vision_head_init', type=str,
                        choices=['trunc_normal', 'small_trunc_normal', 'zero',
                                 'xavier_uniform', 'rare_binary_prior'], default=None,
                        help='initialization for the downstream vision classifier head')
    parser.add_argument('--vision_head_init_std', type=float, default=None,
                        help='override classifier-head truncated-normal weight std')
    parser.add_argument('--vision_squeeze_binary', type=str, default=None,
                        help='return a scalar logit for single-class binary tasks')
    parser.add_argument('--vision_init_head', type=str, default=None,
                        help='whether to initialize the vision classifier head')
    parser.add_argument('--vision_feature_aggregation', type=str, default=None,
                        choices=['gap', 'flatten'],
                        help='feature aggregation used before the vision head')
    parser.add_argument('--eeg_dataset_mean', type=float, default=None,
                        help='training-split global EEG mean in raw clipped units')
    parser.add_argument('--eeg_dataset_std', type=float, default=None,
                        help='training-split global EEG std in raw clipped units')
    parser.add_argument('--eeg_target_std', type=float, default=1.0,
                        help='target std after dataset-level EEG z-score (default: 1)')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA index passed to finetune_main.py')
    parser.add_argument('--device', choices=['cuda', 'cpu', 'auto'], default=None,
                        help='device policy passed to finetune_main.py')
    parser.add_argument('--epochs', type=int, default=None, help='override configured epoch count')
    parser.add_argument('--batch_size', type=int, default=None, help='override configured batch size')
    parser.add_argument('--num_workers', type=int, default=None, help='override configured worker count')
    parser.add_argument('--lr', type=float, default=None, help='override configured learning rate')
    parser.add_argument('--backbone_lr_scale', type=float, default=None,
                        help='backbone LR multiplier when multi_lr is enabled')
    parser.add_argument('--weight_decay', type=float, default=None, help='override configured weight decay')
    parser.add_argument('--clip_value', type=float, default=None,
                        help='gradient-norm clipping threshold; <=0 disables clipping')
    parser.add_argument('--ema_decay', type=float, default=None,
                        help='model EMA decay; 0 disables EMA')
    parser.add_argument('--min_lr', type=float, default=None, help='minimum cosine-decay learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=None, help='linear warmup epoch count')
    parser.add_argument('--warmup_start_factor', type=float, default=None,
                        help='warmup starting LR fraction')
    parser.add_argument('--optimizer', type=str, default=None, help='optimizer passed to finetune_main.py')
    parser.add_argument('--label_smoothing', type=float, default=None,
                        help='label smoothing passed to finetune_main.py')
    parser.add_argument('--binary_pos_weight', type=float, default=None,
                        help='positive-class weight for BCE binary tasks')
    parser.add_argument('--dropout', type=float, default=None, help='dropout passed to finetune_main.py')
    parser.add_argument('--drop_path_rate', type=float, default=None,
                        help='vision-backbone stochastic-depth rate')
    parser.add_argument('--early_stop', type=int, default=None, help='early stop patience passed to finetune_main.py')
    parser.add_argument('--test_each_epoch', type=str, default=None,
                        help='evaluate test after every epoch, true/false')
    parser.add_argument('--run_final_test', type=str, default=None,
                        help='evaluate the selected checkpoint on test, true/false')
    parser.add_argument('--selection_metric', type=str, default=None,
                        choices=['auto', 'pr_auc', 'roc_auc', 'kappa', 'f1', 'ba'],
                        help='validation metric used for checkpoint selection')
    parser.add_argument('--amp', type=str, default=None,
                        help='use CUDA float16 automatic mixed precision, true/false')
    parser.add_argument('--amp_dtype', type=str, default=None,
                        choices=['float16', 'bfloat16'],
                        help='CUDA autocast dtype when AMP is enabled')
    parser.add_argument('--mental_scale', type=float, default=None,
                        help='MentalArithmetic divisor applied after clipping')
    parser.add_argument('--frozen', type=str, default=None, help='whether to freeze backbone, true/false')
    parser.add_argument('--multi_lr', type=str, default=None, help='whether to use multi learning rates, true/false')
    parser.add_argument('--use_pretrained_weights', type=str, default=None,
                        help='whether to use timm/foundation pretrained weights, true/false')
    parser.add_argument('--balanced_sampling', type=str, default=None,
                        help='whether to balance classes with a weighted training sampler, true/false')
    parser.add_argument('--balanced_sampling_power', type=float, default=None,
                        help='class-balance exponent; 1 fully balances, 0 preserves natural sampling')
    parser.add_argument('--balanced_sampling_min_share', type=float, default=None,
                        help='minimum target sampling share for every class; 0 disables')
    parser.add_argument('--mirror_augmentation', type=str, default=None,
                        help='whether to apply train-time left/right channel mirror augmentation, true/false')
    parser.add_argument('--mirror_prob', type=float, default=None,
                        help='probability for train-time channel mirror augmentation')
    parser.add_argument('--time_roll_augmentation', type=str, default=None,
                        help='whether to circular-roll training EEG along time, true/false')
    parser.add_argument('--time_roll_prob', type=float, default=None,
                        help='probability for train-time circular time roll')
    parser.add_argument('--time_roll_max_fraction', type=float, default=None,
                        help='maximum absolute time roll as a fraction of signal length')
    parser.add_argument('--amplitude_scale_augmentation', type=str, default=None,
                        help='whether to randomly scale each training EEG sample, true/false')
    parser.add_argument('--amplitude_scale_prob', type=float, default=None,
                        help='probability for train-time random amplitude scaling')
    parser.add_argument('--amplitude_scale_min', type=float, default=None,
                        help='minimum train-time EEG amplitude scale')
    parser.add_argument('--amplitude_scale_max', type=float, default=None,
                        help='maximum train-time EEG amplitude scale')
    parser.add_argument('--amplitude_scale_distribution', type=str,
                        choices=['log_uniform', 'uniform'], default=None,
                        help='distribution for train-time random amplitude scaling')
    parser.add_argument('--mixup_augmentation', type=str, default=None,
                        help='apply batch-level Mixup during binary training, true/false')
    parser.add_argument('--mixup_prob', type=float, default=None,
                        help='probability of applying Mixup to a training batch')
    parser.add_argument('--mixup_alpha', type=float, default=None,
                        help='symmetric Beta distribution alpha for Mixup')
    parser.add_argument('--shu_clip_limit', type=float, default=None,
                        help='SHU-MI raw-value clip limit before the vision adapter')
    parser.add_argument('--shu_scale', type=float, default=None,
                        help='SHU-MI divisor applied after clipping')
    parser.add_argument('--shu_bandpass_low', type=float, default=None,
                        help='optional SHU-MI band-pass low cutoff in Hz')
    parser.add_argument('--shu_bandpass_high', type=float, default=None,
                        help='optional SHU-MI band-pass high cutoff in Hz')
    parser.add_argument('--shu_filter_order', type=int, default=None,
                        help='Butterworth order for optional SHU-MI band-pass')
    parser.add_argument('--physio_lowpass_hz', type=float, default=None,
                        help='optional PhysioNet-MI low-pass cutoff in Hz')
    parser.add_argument('--physio_filter_order', type=int, default=None,
                        help='Butterworth order for optional PhysioNet-MI low-pass')
    parser.add_argument('--mumtaz_lowpass_hz', type=float, default=None,
                        help='optional Mumtaz2016 low-pass cutoff in Hz')
    parser.add_argument('--mumtaz_filter_order', type=int, default=None,
                        help='Butterworth order for optional Mumtaz2016 low-pass')
    parser.add_argument('--faced_input_norm', type=str, default=None,
                        choices=['clip_scale', 'robust_sample'],
                        help='FACED input normalization')
    parser.add_argument('--faced_robust_clip', type=float, default=None,
                        help='absolute clip after FACED per-trial robust normalization')
    parser.add_argument('--random_init', action='store_true',
                        help='set --use_pretrained_weights False for offline/debug runs')
    parser.add_argument('--online_weights', action='store_true',
                        help='allow Hugging Face network checks/downloads for timm pretrained weights')
    parser.add_argument('--continue_on_error', action='store_true', help='continue after a failed dataset')
    args, extra_args = parser.parse_known_args()

    experiment = load_experiment_config(args.config) if args.config else None
    if experiment:
        if args.all:
            raise SystemExit('--all cannot be combined with --config; list datasets in the YAML instead')
        apply_experiment_config(args, experiment)

    if args.backbone_config:
        profile = load_backbone_config(
            args.backbone_config,
            dataset=args.dataset[0] if args.dataset else None,
        )
        if args.backbone_name is None:
            args.backbone_name = backbone_name_for(profile)

    if args.list:
        list_experiments(args)
        return

    names = selected_names(args)
    if args.check_only:
        ok = check_experiments(names)
        raise SystemExit(0 if ok else 1)

    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = (experiment or {}).get('protocol', {}).get('seeds', [None])
    for seed in seeds:
        for name in names:
            command = build_command(name, args, extra_args, seed=seed)
            log_path = build_log_path(name, args, seed=seed)
            print_command(name, command, log_path)
            returncode = run_command(command, args, log_path)
            if returncode and not args.continue_on_error:
                raise SystemExit(returncode)


def selected_names(args):
    if args.all:
        return list(EXPERIMENTS.keys())
    if args.dataset:
        return args.dataset
    raise SystemExit('Please pass --dataset DATASET, --all, --list, or --check_only --all.')


def build_command(name, args, extra_args=None, seed=None):
    cfg = EXPERIMENTS[name]
    training = training_config_for(
        name,
        model_arch=args.model_arch,
        backbone_name=args.backbone_name,
        backbone_config=args.backbone_config,
    )
    model_root = format_output_root(args.model_root, args, name, seed)
    model_dir = Path(model_root) / safe_name(name)
    device = args.device
    if device is None and args.dry_run:
        device = 'auto'

    command = [
        args.python,
        str(ROOT / 'finetune_main.py'),
        '--downstream_dataset', name,
        '--model_arch', args.model_arch,
        '--num_of_classes', str(cfg['classes']),
        '--model_dir', str(model_dir),
        '--num_workers', str(configured_value(args, training, 'num_workers')),
        '--datasets_dir', cfg['datasets_dir'],
        '--cuda', str(args.cuda),
        '--lr', str(configured_value(args, training, 'lr')),
        '--batch_size', str(configured_value(args, training, 'batch_size')),
        '--epochs', str(configured_value(args, training, 'epochs')),
        '--weight_decay', str(configured_value(args, training, 'weight_decay')),
        '--clip_value', str(configured_value(args, training, 'clip_value')),
    ]
    append_optional_training_args(command, args, training)
    if args.backbone_name:
        command.extend(['--backbone_name', args.backbone_name])
    if args.backbone_config:
        command.extend(['--backbone_config', str(args.backbone_config)])
    if args.vision_fold_factor is not None:
        command.extend(['--vision_fold_factor', str(args.vision_fold_factor)])
    if args.vision_channel_repeat is not None:
        command.extend(['--vision_channel_repeat', str(args.vision_channel_repeat)])
    if args.vision_height_stride is not None:
        command.extend(['--vision_height_stride', str(args.vision_height_stride)])
    if args.vision_no_pad:
        command.extend(['--vision_no_pad', 'true'])
    if args.vision_head_init is not None:
        command.extend(['--vision_head_init', args.vision_head_init])
    if args.vision_head_init_std is not None:
        command.extend(['--vision_head_init_std', str(args.vision_head_init_std)])
    if args.vision_squeeze_binary is not None:
        command.extend(['--vision_squeeze_binary', str(args.vision_squeeze_binary)])
    if args.vision_init_head is not None:
        command.extend(['--vision_init_head', str(args.vision_init_head)])
    if args.vision_feature_aggregation is not None:
        command.extend(['--vision_feature_aggregation', str(args.vision_feature_aggregation)])
    if args.eeg_dataset_mean is not None:
        # Use --key=value so a negative mean in scientific notation is not
        # mistaken for another argparse option by the child process.
        command.append('--eeg_dataset_mean={}'.format(args.eeg_dataset_mean))
    if args.eeg_dataset_std is not None:
        command.extend(['--eeg_dataset_std', str(args.eeg_dataset_std)])
    if args.eeg_target_std != 1.0:
        command.extend(['--eeg_target_std', str(args.eeg_target_std)])
    if device:
        command.extend(['--device', device])
    if args.dry_run:
        command.append('--dry_run')
    if args.random_init:
        command.extend(['--use_pretrained_weights', 'False'])
    if extra_args:
        command.extend(extra_args)
    if seed is not None:
        command.extend(['--seed', str(seed)])
    return command


def append_optional_training_args(command, args, training):
    for attr in TRAINING_KEYS:
        if attr in {'lr', 'batch_size', 'epochs', 'weight_decay', 'clip_value', 'num_workers'}:
            continue
        if attr == 'use_pretrained_weights' and args.random_init:
            continue
        value = configured_value(args, training, attr)
        if value is not None:
            command.extend(['--{}'.format(attr), str(value)])


def run_env(args):
    env = os.environ.copy()
    if not args.online_weights:
        env.setdefault('HF_HUB_OFFLINE', '1')
    return env


def build_log_path(name, args, seed=None):
    if args.no_log_file:
        return None
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = '{}_{}_pid{}.log'.format(timestamp, safe_name(name), os.getpid())
    log_root = format_output_root(args.log_root, args, name, seed)
    return Path(log_root) / args.model_arch / safe_name(name) / filename


def format_output_root(root, args, dataset, seed):
    value = str(root)
    if seed is not None and '{seed}' not in value and getattr(args, 'config', None):
        value = str(Path(value) / 'seed{}'.format(seed))
    return value.format(seed='' if seed is None else seed, dataset=safe_name(dataset))


def apply_experiment_config(args, config):
    """Apply YAML values only when the corresponding CLI flag was omitted."""
    explicit = {item.split('=', 1)[0] for item in sys.argv[1:] if item.startswith('--')}

    def set_if_absent(attr, value, flag=None):
        if value is not None and ('--{}'.format(flag or attr) not in explicit):
            setattr(args, attr, value)

    datasets = config['dataset']
    if args.dataset is None:
        args.dataset = datasets
    elif any(dataset not in datasets for dataset in args.dataset):
        raise SystemExit(
            '--dataset values must be listed by experiment config {}; got {}'.format(
                ', '.join(datasets), ', '.join(args.dataset)
            )
        )
    set_if_absent('model_arch', config.get('model_arch'), 'model_arch')
    set_if_absent('backbone_name', config.get('backbone_name') or config.get('backbone', {}).get('name'), 'backbone_name')
    set_if_absent('backbone_config', config.get('backbone_config'), 'backbone_config')
    vision = config.get('vision', {})
    for key, attr in {
        'fold_factor': 'vision_fold_factor', 'channel_repeat': 'vision_channel_repeat',
        'height_stride': 'vision_height_stride', 'no_pad': 'vision_no_pad',
        'head_init': 'vision_head_init', 'head_init_std': 'vision_head_init_std',
        'squeeze_binary': 'vision_squeeze_binary', 'init_head': 'vision_init_head',
        'feature_aggregation': 'vision_feature_aggregation',
    }.items():
        set_if_absent(attr, vision.get(key), attr)
    adapter = vision.get('adapter', {})
    if isinstance(adapter, dict):
        set_if_absent('vision_fold_factor', adapter.get('fold_factor'), 'vision_fold_factor')
    for key, value in config.get('training', {}).items():
        if hasattr(args, key):
            set_if_absent(key, value, key)
    runtime = config.get('output', {})
    for key in ('model_root', 'log_root'):
        set_if_absent(key, runtime.get(key), key)
    set_if_absent('cuda', config.get('cuda'), 'cuda')
    set_if_absent('device', config.get('device'), 'device')
    args.config = config.get('_path', args.config)


def run_command(command, args, log_path):
    if log_path is None:
        return subprocess.run(command, cwd=ROOT, env=run_env(args)).returncode

    log_path = (ROOT / log_path).resolve() if not log_path.is_absolute() else log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', encoding='utf-8', errors='replace') as log_file:
        log_file.write('{}\n'.format(' '.join(command)))
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=run_env(args),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            errors='replace',
        )
        for line in process.stdout:
            print(line, end='')
            log_file.write(line)
            log_file.flush()
        return process.wait()


def configured_value(args, training, key):
    value = getattr(args, key, None)
    if value is not None:
        return value
    return training.get(key)


def safe_name(name):
    return name.lower().replace('-', '_').replace(' ', '_')


def vision_adapter_summary(cfg):
    adapter = cfg.get('vision', {}).get('adapter', {})
    if not adapter:
        return '-'
    fold_factor = getattr(vision_adapter_summary, 'fold_factor_override', None)
    return 'phase_fold,P={}'.format(fold_factor or adapter['fold_factor'])


def list_experiments(args):
    vision_adapter_summary.fold_factor_override = args.vision_fold_factor
    print('{:<18} {:<8} {:<11} {:<7} {:<18} {:<6} {:<6} {:<9} {:<9} {:<20} {}'.format(
        'dataset',
        'status',
        'task',
        'classes',
        'model',
        'epoch',
        'batch',
        'lr',
        'wd',
        'vision_adapter',
        'path',
    ))
    for name, cfg in EXPERIMENTS.items():
        training = training_config_for(
            name,
            model_arch=args.model_arch,
            backbone_name=args.backbone_name,
            backbone_config=args.backbone_config,
        )
        print('{:<18} {:<8} {:<11} {:<7} {:<18} {:<6} {:<6} {:<9} {:<9} {:<20} {}'.format(
            name,
            'RERUN' if name in RERUN_REQUIRED_DATASETS else '-',
            cfg['task'],
            cfg['classes'],
            model_summary(cfg, args),
            training['epochs'],
            training['batch_size'],
            training['lr'],
            training['weight_decay'],
            vision_adapter_summary(cfg),
            cfg['datasets_dir'],
        ))


def model_summary(cfg, args):
    if args.model_arch != 'vision':
        return args.model_arch
    return args.backbone_name or cfg.get('vision', {}).get('backbone_name', 'vision')


def check_experiments(names):
    ok = True
    lmdb_available = importlib.util.find_spec('lmdb') is not None
    print('lmdb: {}'.format('available' if lmdb_available else 'missing'))
    for name in names:
        cfg = EXPERIMENTS[name]
        path = (ROOT / cfg['datasets_dir']).resolve()
        exists = path.exists()
        ok = ok and exists
        details = 'missing'
        if exists:
            details = describe_dataset(path, cfg, lmdb_available)
            if cfg['storage'] == 'lmdb' and not lmdb_available:
                ok = False
        print('{:<18} {:<7} {}'.format(name, 'ok' if exists else 'missing', details))
    return ok


def describe_dataset(path, cfg, lmdb_available):
    if cfg['storage'] == 'pkl_split':
        counts = {}
        for split, dirname in cfg['split_dirs'].items():
            split_path = path / dirname
            counts[split] = len(os.listdir(split_path)) if split_path.exists() else 'missing'
        return format_counts(counts)

    if cfg['storage'] == 'isruc_npy':
        return format_counts(count_isruc(path))

    if cfg['storage'] == 'lmdb':
        if not (path / 'data.mdb').exists():
            return 'data.mdb missing'
        if not lmdb_available:
            return 'data.mdb found; install lmdb to read split counts'
        return format_counts(count_lmdb(path))

    return 'unknown storage {}'.format(cfg['storage'])


def count_isruc(path):
    seq_root = path / 'seq'
    counts = {'train': 0, 'val': 0, 'test': 0}
    for subject_idx in range(1, 101):
        subject_dir = seq_root / 'ISRUC-group1-{}'.format(subject_idx)
        count = len(os.listdir(subject_dir)) if subject_dir.exists() else 0
        if subject_idx <= 80:
            counts['train'] += count
        elif subject_idx <= 90:
            counts['val'] += count
        else:
            counts['test'] += count
    return counts


def count_lmdb(path):
    import lmdb

    env = lmdb.open(str(path), readonly=True, lock=False, readahead=True, meminit=False)
    with env.begin(write=False) as txn:
        keys = pickle.loads(txn.get('__keys__'.encode()))
    return {split: len(keys.get(split, [])) for split in ('train', 'val', 'test')}


def format_counts(counts):
    return ', '.join('{}={}'.format(split, counts[split]) for split in ('train', 'val', 'test'))


def print_command(name, command, log_path=None):
    print('\n[{}] {}'.format(name, ' '.join(command)), flush=True)
    if log_path is not None:
        print('[{}] log: {}'.format(name, log_path), flush=True)


if __name__ == '__main__':
    main()
