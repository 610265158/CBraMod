import argparse
import importlib
import random
from pathlib import Path

import numpy as np
import torch

from configs.downstream import DOWNSTREAM_11_CONFIGS
from configs.downstream import TRAINING_KEYS
from configs.downstream import dataset_registry
from configs.downstream import training_config_for
from finetune_evaluator import Evaluator
from finetune_trainer import Trainer


DATASET_REGISTRY = dataset_registry()
DATASET_REGISTRY.update({
    'SEED-VIG': {
        'dataset_module': 'datasets.seedvig_dataset',
        'model_module': 'models.legacy.model_for_seedvig',
        'task': 'regression',
    },
    'BCIC-IV-2a': {
        'dataset_module': 'datasets.bciciv2a_dataset',
        'model_module': 'models.legacy.model_for_bciciv2a',
        'task': 'multiclass',
    },
})

TRAIN_METHODS = {
    'binary': 'train_for_binaryclass',
    'multiclass': 'train_for_multiclass',
    'regression': 'train_for_regression',
}

MODEL_MODULES = {
    'vision': 'models.vision_model',
    'eegnet': 'models.eegnet_model',
}


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('yes', 'true', 't', '1', 'y'):
        return True
    if value in ('no', 'false', 'f', '0', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser(description='Big model downstream')
    parser.add_argument('--seed', type=int, default=3407, help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default=1, help='cuda number (default: 1)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu', 'auto'],
                        help='device policy: cuda requires CUDA, auto falls back to CPU')
    parser.add_argument('--epochs', type=int, default=None, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='batch size for training')
    parser.add_argument('--lr', type=float, default=None, help='learning rate')
    parser.add_argument('--backbone_lr_scale', type=float, default=None,
                        help='backbone LR multiplier when --multi_lr is enabled')
    parser.add_argument('--weight_decay', type=float, default=None, help='weight decay')
    parser.add_argument('--min_lr', type=float, default=None,
                        help='minimum LR for cosine decay')
    parser.add_argument('--warmup_epochs', type=int, default=None,
                        help='linear LR warmup epochs')
    parser.add_argument('--warmup_start_factor', type=float, default=None,
                        help='warmup starting LR as a fraction of each group LR')
    parser.add_argument('--optimizer', type=str, default=None, help='optimizer (AdamW, SGD)')
    parser.add_argument('--clip_value', type=float, default=None, help='clip_value')
    parser.add_argument('--dropout', type=float, default=None, help='dropout')
    parser.add_argument('--drop_path_rate', type=float, default=None,
                        help='stochastic-depth rate passed to the vision backbone')
    parser.add_argument('--model_arch', type=str, default='vision', choices=sorted(MODEL_MODULES),
                        help='downstream model architecture')
    parser.add_argument('--backbone_name', type=str, default=None,
                        help='override timm backbone name for --model_arch vision')
    parser.add_argument('--vision_fold_factor', type=int, default=None,
                        help='override phase-interleaved temporal fold factor P (minimum: 1)')
    parser.add_argument('--vision_channel_repeat', type=int, default=1,
                        help='repeat each EEG channel consecutively before the vision adapter')
    parser.add_argument('--vision_height_stride', type=int, default=32,
                        choices=[1, 2, 4, 8, 16, 32],
                        help='target CNN output stride along EEG-channel height; time stride is unchanged')
    parser.add_argument('--vision_head_init', type=str, default='trunc_normal',
                        choices=['trunc_normal', 'zero', 'xavier_uniform'],
                        help='initialization for the downstream vision classifier head')
    parser.add_argument('--eeg_dataset_mean', type=float, default=None,
                        help='training-split global EEG mean in raw clipped units')
    parser.add_argument('--eeg_dataset_std', type=float, default=None,
                        help='training-split global EEG std in raw clipped units')
    parser.add_argument('--eeg_target_std', type=float, default=1.0,
                        help='target std after dataset-level EEG z-score (default: 1)')
    parser.add_argument('--shu_clip_limit', type=float, default=512.0,
                        help='SHU-MI raw-value clip limit before the vision adapter')
    parser.add_argument('--shu_scale', type=float, default=64.0,
                        help='SHU-MI divisor applied after clipping (default: 64)')
    parser.add_argument('--faced_input_norm', type=str, default=None,
                        choices=['clip_scale', 'robust_sample'],
                        help='FACED input normalization; robust_sample uses one median/MAD per trial')
    parser.add_argument('--faced_robust_clip', type=float, default=None,
                        help='absolute clip after FACED per-trial robust normalization')
    parser.add_argument('--classifier', type=str, default='all_patch_reps',
                        help='[all_patch_reps, all_patch_reps_twolayer, '
                             'all_patch_reps_onelayer, avgpooling_patch_reps]')
    # all_patch_reps: use all patch features with a three-layer classifier;
    # all_patch_reps_twolayer: use all patch features with a two-layer classifier;
    # all_patch_reps_onelayer: use all patch features with a one-layer classifier;
    # avgpooling_patch_reps: use average pooling for patch features;

    """############ Downstream dataset settings ############"""
    parser.add_argument('--downstream_dataset', type=str, default='FACED',
                        choices=sorted(DATASET_REGISTRY.keys()),
                        help='downstream dataset name')
    parser.add_argument('--datasets_dir', type=str, default=None, help='datasets_dir')
    parser.add_argument('--num_of_classes', type=int, default=None, help='number of classes')
    parser.add_argument('--model_dir', type=str, default=None, help='model_dir')
    """############ Downstream dataset settings ############"""

    parser.add_argument('--num_workers', type=int, default=None, help='num_workers')
    parser.add_argument('--label_smoothing', type=float, default=None, help='label_smoothing')
    parser.add_argument('--multi_lr', type=str2bool, default=None,
                        help='multi_lr')  # set different learning rates for different modules
    parser.add_argument('--frozen', type=str2bool,
                        default=None, help='frozen')
    parser.add_argument('--use_pretrained_weights', type=str2bool,
                        default=None, help='use_pretrained_weights')
    parser.add_argument('--balanced_sampling', type=str2bool,
                        default=None, help='balance classes with a weighted sampler on the training split')
    parser.add_argument('--mirror_augmentation', type=str2bool,
                        default=None, help='randomly mirror left/right EEG channels on the training split')
    parser.add_argument('--mirror_prob', type=float,
                        default=None, help='probability for train-time channel mirror augmentation')
    parser.add_argument('--time_roll_augmentation', type=str2bool,
                        default=None, help='randomly circular-roll training EEG along time')
    parser.add_argument('--time_roll_prob', type=float,
                        default=None, help='probability for train-time circular time roll')
    parser.add_argument('--time_roll_max_fraction', type=float,
                        default=None, help='maximum absolute time roll as a fraction of signal length')
    parser.add_argument('--amplitude_scale_augmentation', type=str2bool,
                        default=None, help='randomly scale each training EEG sample amplitude')
    parser.add_argument('--amplitude_scale_prob', type=float,
                        default=None, help='probability for train-time random amplitude scaling')
    parser.add_argument('--amplitude_scale_min', type=float,
                        default=None, help='minimum train-time EEG amplitude scale')
    parser.add_argument('--amplitude_scale_max', type=float,
                        default=None, help='maximum train-time EEG amplitude scale')
    parser.add_argument('--foundation_dir', type=str,
                        default='pretrained_weights/pretrained_weights.pth',
                        help='foundation_dir')
    parser.add_argument('--vision_pretrained_checkpoint', type=str, default=None,
                        help='EEG-Vision backbone produced by pretrain_main.py')
    parser.add_argument('--early_stop', type=int,
                        default=None,
                        help='early_stop')
    parser.add_argument('--test_each_epoch', type=str2bool, default=True,
                        help='evaluate the test split after every epoch; disable for formal runs')
    parser.add_argument('--run_final_test', type=str2bool, default=True,
                        help='evaluate the best checkpoint on test after training')
    parser.add_argument('--selection_metric', type=str, default='auto',
                        choices=['auto', 'pr_auc', 'roc_auc', 'kappa', 'f1', 'ba'],
                        help='validation metric used to select the best checkpoint')
    parser.add_argument('--amp', type=str2bool, default=None,
                        help='use CUDA float16 automatic mixed precision during training')
    parser.add_argument('--amp_dtype', type=str, default=None,
                        choices=['float16', 'bfloat16'],
                        help='CUDA autocast dtype when AMP is enabled')
    parser.add_argument('--dry_run', action='store_true',
                        help='load one training batch and run one forward pass without training')
    parser.add_argument('--evaluate_checkpoint', type=str, default=None,
                        help='load a saved model state and evaluate only on the test split')
    params = parser.parse_args()
    apply_downstream_defaults(params)

    params.device = resolve_device(params.device, params.cuda)
    print(params)

    setup_seed(params.seed)
    print('The downstream dataset is {}'.format(params.downstream_dataset))

    registry = DATASET_REGISTRY[params.downstream_dataset]
    params.downstream_task = registry['task']
    dataset_module = import_selected_module(registry['dataset_module'])
    model_module_name = registry.get('model_module', MODEL_MODULES[params.model_arch])
    model_module = import_selected_module(model_module_name)
    load_dataset = dataset_module.LoadDataset(params)
    data_loader = load_dataset.get_data_loader()
    model = model_module.Model(params)

    if params.dry_run:
        dry_run(params, data_loader, model)
        print('Dry run done!!!!!')
        return

    if params.evaluate_checkpoint:
        evaluate_checkpoint(params, data_loader, model)
        print('Checkpoint evaluation done!!!!!')
        return

    trainer = Trainer(params, data_loader, model)
    train_method = getattr(trainer, TRAIN_METHODS[registry['task']])
    train_method()
    print('Done!!!!!')


def apply_downstream_defaults(params):
    cfg = DOWNSTREAM_11_CONFIGS.get(params.downstream_dataset)
    if cfg:
        training = training_config_for(
            params.downstream_dataset,
            model_arch=params.model_arch,
            backbone_name=params.backbone_name,
        )
    else:
        training = legacy_training_defaults()

    for key in TRAINING_KEYS:
        if getattr(params, key, None) is None:
            setattr(params, key, training[key])

    if cfg is not None:
        if params.datasets_dir is None:
            params.datasets_dir = cfg['datasets_dir']
        if params.num_of_classes is None:
            params.num_of_classes = cfg['classes']
    else:
        if params.num_of_classes is None:
            params.num_of_classes = 9

    if params.model_dir is None:
        params.model_dir = str(Path('experiments/checkpoints/manual') / safe_name(params.downstream_dataset))


def legacy_training_defaults():
    return {
        'lr': 0.0005,
        'backbone_lr_scale': 0.1,
        'batch_size': 64,
        'epochs': 50,
        'weight_decay': 5e-2,
        'min_lr': 1e-6,
        'warmup_epochs': 0,
        'warmup_start_factor': 0.1,
        'clip_value': 1,
        'num_workers': 16,
        'optimizer': 'AdamW',
        'label_smoothing': 0.1,
        'dropout': 0.1,
        'drop_path_rate': 0.0,
        'early_stop': 10,
        'frozen': False,
        'multi_lr': True,
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


def safe_name(name):
    return name.lower().replace('-', '_').replace(' ', '_')


def import_selected_module(module_name):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == 'lmdb':
            raise ModuleNotFoundError(
                'The selected dataset loader requires lmdb. Install it in the active environment, '
                'for example: pip install lmdb'
            ) from exc
        raise


def resolve_device(device_policy, cuda_index):
    if device_policy == 'cpu':
        return 'cpu'
    if device_policy == 'auto' and not torch.cuda.is_available():
        print('CUDA is not available; using CPU because --device auto was set.')
        return 'cpu'
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available. Use a CUDA node, or pass --device cpu for dry runs.')
    torch.cuda.set_device(cuda_index)
    return 'cuda:{}'.format(cuda_index)


def dry_run(params, data_loader, model):
    device = torch.device(params.device)
    model = model.to(device)
    model.eval()
    x, y = next(iter(data_loader['train']))
    x = x.to(device)
    with torch.no_grad():
        pred = model(x)
    print('Dry run batch x: {}, y: {}, pred: {}'.format(tuple(x.shape), tuple(y.shape), tuple(pred.shape)))


def evaluate_checkpoint(params, data_loader, model):
    """Evaluate a saved downstream state without constructing a trainer/optimizer."""
    device = torch.device(params.device)
    model = model.to(device)
    model.eval()

    checkpoint = torch.load(params.evaluate_checkpoint, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        checkpoint = checkpoint['model_state_dict']
    model.load_state_dict(checkpoint, strict=True)
    evaluator = Evaluator(params, data_loader['test'])
    with torch.no_grad():
        if params.downstream_task == 'binary':
            ba, pr_auc, roc_auc, cm = evaluator.get_metrics_for_binaryclass(model)
            print('Checkpoint Test Evaluation: ba: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}'.format(
                ba, pr_auc, roc_auc
            ))
        elif params.downstream_task == 'multiclass':
            ba, kappa, f1, cm = evaluator.get_metrics_for_multiclass(model)
            print('Checkpoint Test Evaluation: ba: {:.5f}, kappa: {:.5f}, f1: {:.5f}'.format(
                ba, kappa, f1
            ))
        else:
            corrcoef, r2, rmse = evaluator.get_metrics_for_regression(model)
            print('Checkpoint Test Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}'.format(
                corrcoef, r2, rmse
            ))
        print(cm)


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()
