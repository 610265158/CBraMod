import copy
import os
import random
from contextlib import contextmanager, nullcontext
from timeit import default_timer as timer

import numpy as np
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from timm.utils import ModelEmaV2
from tqdm import tqdm

from finetune_evaluator import Evaluator

import torch.nn as nn


@contextmanager
def preserve_random_state():
    """Keep diagnostic evaluation from changing the next epoch's training RNG."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = None
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        cuda_states = torch.cuda.get_rng_state_all()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

class Trainer(object):
    def __init__(self, params, data_loader, model):
        self.params = params
        self.data_loader = data_loader

        self.val_eval = Evaluator(params, self.data_loader['val'])
        self.test_eval = Evaluator(params, self.data_loader['test'])

        self.device = torch.device(getattr(params, 'device', 'cuda'))
        self.use_amp = self.device.type == 'cuda' and getattr(params, 'amp', True)
        amp_dtypes = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
        }
        self.amp_dtype_name = getattr(params, 'amp_dtype', 'float16')
        self.amp_dtype = amp_dtypes[self.amp_dtype_name]
        self.model = model.to(self.device)
        self.ema_decay = float(getattr(params, 'ema_decay', 0.0))
        if self.ema_decay < 0 or self.ema_decay >= 1:
            raise ValueError('--ema_decay must be 0 (disabled) or in the interval (0, 1)')
        self.ema_model = None
        if self.ema_decay > 0:
            self.ema_model = ModelEmaV2(self.model, decay=self.ema_decay)
            self.ema_model.requires_grad_(False)
            print('timm ModelEmaV2 enabled: decay={}'.format(self.ema_decay))
        if self.params.downstream_dataset in ['FACED', 'SEED-V', 'PhysioNet-MI', 'ISRUC', 'BCIC2020-3', 'TUEV', 'BCIC-IV-2a']:
            self.criterion = CrossEntropyLoss(label_smoothing=self.params.label_smoothing).to(self.device)
        elif self.params.downstream_dataset in ['SHU-MI', 'CHB-MIT', 'Mumtaz2016', 'MentalArithmetic', 'TUAB']:
            binary_pos_weight = float(getattr(self.params, 'binary_pos_weight', 1.0))
            if binary_pos_weight <= 0:
                raise ValueError('--binary_pos_weight must be greater than 0')
            pos_weight = torch.tensor(binary_pos_weight, dtype=torch.float32, device=self.device)
            self.criterion = BCEWithLogitsLoss(pos_weight=pos_weight).to(self.device)
            if binary_pos_weight != 1.0:
                print('Weighted BCE enabled: pos_weight={}'.format(binary_pos_weight))
        elif self.params.downstream_dataset == 'SEED-VIG':
            self.criterion = MSELoss().to(self.device)

        self.best_model_states = None

        self.optimizer = self.configure_optimizers()

        self.data_length = len(self.data_loader['train'])
        self.optimizer_scheduler = self.configure_scheduler()
        print(self.model)
        self.scaler = torch.amp.GradScaler(
            'cuda',
            enabled=self.use_amp and self.amp_dtype == torch.float16,
        )

    def save_best_model_state(self):
        os.makedirs(self.params.model_dir, exist_ok=True)
        model_path = os.path.join(self.params.model_dir, "best.pth")
        state = self.best_model_states
        if state is None:
            state = self.evaluation_model().state_dict()
        torch.save(state, model_path)
        print("best model save in " + model_path)

    def evaluation_model(self):
        return self.ema_model.module if self.ema_model is not None else self.model

    def update_ema(self):
        if self.ema_model is not None:
            self.ema_model.update(self.model)

    def amp_context(self):
        if self.use_amp:
            return torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype)
        return nullcontext()

    def configure_optimizers(self, ):
            """
            This long function is unfortunately doing something very simple and is being very defensive:
            We are separating out all parameters of the model into two buckets: those that will experience
            weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
            We are then returning the PyTorch optimizer object.
            """

            # separate out all parameters to those that will and won't experience regularizing weight decay

            use_multi_lr = bool(self.params.multi_lr) and hasattr(self.model, 'backbone')
            backbone_scale = self.params.backbone_lr_scale if use_multi_lr else 1.0
            if not 0 < backbone_scale <= 1:
                raise ValueError('--backbone_lr_scale must be in (0, 1]')

            buckets = {}
            for name, param in self.model.named_parameters():
                if getattr(self.params, 'verbose_optimizer', False):
                    print('checking {}'.format(name))
                if not param.requires_grad:
                    continue
                role = 'backbone' if use_multi_lr and name.startswith('backbone.') else 'head'
                decay = param.ndim > 1 and not name.endswith('.bias')
                buckets.setdefault((role, decay), []).append(param)

            role_lrs = {
                'backbone': self.params.lr * backbone_scale,
                'head': self.params.lr,
            }
            optim_groups = []
            for (role, decay), parameters in buckets.items():
                optim_groups.append({
                    'params': parameters,
                    'lr': role_lrs[role],
                    'weight_decay': self.params.weight_decay if decay else 0.0,
                    'group_name': '{}_{}'.format(role, 'decay' if decay else 'no_decay'),
                })
            optimizer_name = self.params.optimizer.lower()
            if optimizer_name == 'adamw':
                optimizer = torch.optim.AdamW(optim_groups, self.params.lr)
            elif optimizer_name == 'adam':
                optimizer = torch.optim.Adam(optim_groups, self.params.lr)
            elif optimizer_name == 'sgd':
                optimizer = torch.optim.SGD(optim_groups, self.params.lr, momentum=0.9)
            else:
                raise ValueError('Unsupported optimizer: {}'.format(self.params.optimizer))
            print(
                'Optimizer groups: {}'.format([
                    {
                        'name': group['group_name'],
                        'lr': group['lr'],
                        'weight_decay': group['weight_decay'],
                        'tensors': len(group['params']),
                    }
                    for group in optim_groups
                ])
            )
            return optimizer

    def configure_scheduler(self):
        warmup_epochs = self.params.warmup_epochs
        if warmup_epochs < 0 or warmup_epochs >= self.params.epochs:
            raise ValueError('--warmup_epochs must be >= 0 and smaller than --epochs')
        if not 0 < self.params.warmup_start_factor <= 1:
            raise ValueError('--warmup_start_factor must be in (0, 1]')

        cosine_epochs = self.params.epochs - warmup_epochs
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cosine_epochs,
            eta_min=self.params.min_lr,
        )
        if warmup_epochs == 0:
            return cosine

        warmup = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=self.params.warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )

    def train_for_multiclass(self):
        f1_best = 0
        kappa_best = float('-inf')
        ba_best = 0
        selection_best = float('-inf')
        cm_best = None
        best_f1_epoch = 0
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            loss_sum = torch.zeros((), device=self.device)
            for batch_index, (x, y) in enumerate(tqdm(self.data_loader['train'], mininterval=10)):
                self.optimizer.zero_grad(set_to_none=True)
                x = x.to(self.device, non_blocking=self.use_amp)
                y = y.to(self.device, non_blocking=self.use_amp)
                with self.amp_context():
                    pred = self.model(x)
                    if self.params.downstream_dataset == 'ISRUC':
                        loss = self.criterion(pred.transpose(1, 2), y)
                    else:
                        loss = self.criterion(pred, y)

                self.ensure_finite_loss(loss, epoch, batch_index)
                loss_sum += loss.detach()
                self.scaler.scale(loss).backward()

                if self.params.clip_value > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.update_ema()
            self.optimizer_scheduler.step()

            mean_loss = (loss_sum / self.data_length).item()
            current_lr = self.optimizer.param_groups[0]['lr']

            with torch.no_grad():
                eval_model = self.evaluation_model()
                ba, kappa, f1, cm = self.val_eval.get_metrics_for_multiclass(eval_model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, ba: {:.5f}, kappa: {:.5f}, f1: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        mean_loss,
                        ba,
                        kappa,
                        f1,
                        current_lr,
                        (timer() - start_time) / 60
                    )
                )
                print(cm)
                selection_score = self.multiclass_selection_score(ba, kappa, f1)
                if selection_score > selection_best:
                    selection_best = selection_score
                    print("{} increasing....saving weights !! ".format(
                        self.resolved_selection_metric('multiclass')
                    ))
                    print("Val Evaluation: ba: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                        ba,
                        kappa,
                        f1,
                    ))
                    best_f1_epoch = epoch + 1
                    ba_best = ba
                    kappa_best = kappa
                    f1_best = f1
                    cm_best = cm
                    self.best_model_states = copy.deepcopy(eval_model.state_dict())
                    self.save_best_model_state()
                if self.params.test_each_epoch:
                    with preserve_random_state():
                        ba, kappa, f1, cm = self.test_eval.get_metrics_for_multiclass(eval_model)
                    print("***************************Test results************************")
                    print(
                        "Test Evaluation: ba: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                            ba,
                            kappa,
                            f1,
                        )
                    )
                    print(cm)
        self.model.load_state_dict(self.best_model_states)
        if not self.params.run_final_test:
            print(
                "Best Val Summary: ba: {:.5f}, kappa: {:.5f}, f1: {:.5f}, "
                "selection_metric: {}, selection_score: {:.5f}".format(
                    ba_best,
                    kappa_best,
                    f1_best,
                    self.resolved_selection_metric('multiclass'),
                    selection_best,
                )
            )
            return
        with torch.no_grad():
            print("***************************Test************************")
            ba, kappa, f1, cm = self.test_eval.get_metrics_for_multiclass(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: ba: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                    ba,
                    kappa,
                    f1,
                )
            )
            print(cm)
            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_ba_{:.5f}_kappa_{:.5f}_f1_{:.5f}.pth".format(best_f1_epoch, ba, kappa, f1)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)

    def train_for_binaryclass(self):
        ba_best = 0
        roc_auc_best = float('-inf')
        pr_auc_best = 0
        selection_best = float('-inf')
        cm_best = None
        best_f1_epoch = 0
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            loss_sum = torch.zeros((), device=self.device)
            for batch_index, (x, y) in enumerate(tqdm(self.data_loader['train'], mininterval=10)):
                self.optimizer.zero_grad(set_to_none=True)
                x = x.to(self.device, non_blocking=self.use_amp)
                y = y.to(self.device, non_blocking=self.use_amp)
                with self.amp_context():
                    pred = self.model(x)

                    loss = self.criterion(pred, y)

                self.ensure_finite_loss(loss, epoch, batch_index)
                loss_sum += loss.detach()

                self.scaler.scale(loss).backward()

                if self.params.clip_value > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.update_ema()
            self.optimizer_scheduler.step()

            mean_loss = (loss_sum / self.data_length).item()
            current_lr = self.optimizer.param_groups[0]['lr']

            with torch.no_grad():
                eval_model = self.evaluation_model()
                ba, pr_auc, roc_auc, cm = self.val_eval.get_metrics_for_binaryclass(eval_model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, ba: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        mean_loss,
                        ba,
                        pr_auc,
                        roc_auc,
                        current_lr,
                        (timer() - start_time) / 60
                    )
                )
                print(cm)
                selection_score = self.binary_selection_score(ba, pr_auc, roc_auc)
                if selection_score > selection_best:
                    selection_best = selection_score
                    print("{} increasing....saving weights !! ".format(
                        self.resolved_selection_metric('binary')
                    ))
                    print("Val Evaluation: ba: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
                        ba,
                        pr_auc,
                        roc_auc,
                    ))
                    best_f1_epoch = epoch + 1
                    ba_best = ba
                    pr_auc_best = pr_auc
                    roc_auc_best = roc_auc
                    cm_best = cm
                    self.best_model_states = copy.deepcopy(eval_model.state_dict())
                    self.save_best_model_state()
                if self.params.test_each_epoch:
                    with preserve_random_state():
                        ba, pr_auc, roc_auc, cm = self.test_eval.get_metrics_for_binaryclass(eval_model)
                    print("***************************Test results************************")
                    print(
                        "Test Evaluation: ba: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
                            ba,
                            pr_auc,
                            roc_auc,
                        )
                    )
                    print(cm)
        self.model.load_state_dict(self.best_model_states)
        if not self.params.run_final_test:
            print(
                "Best Val Summary: ba: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}, "
                "selection_metric: {}, selection_score: {:.5f}".format(
                    ba_best,
                    pr_auc_best,
                    roc_auc_best,
                    self.resolved_selection_metric('binary'),
                    selection_best,
                )
            )
            return
        with torch.no_grad():
            print("***************************Test************************")
            ba, pr_auc, roc_auc, cm = self.test_eval.get_metrics_for_binaryclass(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: ba: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
                    ba,
                    pr_auc,
                    roc_auc,
                )
            )
            print(cm)
            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_ba_{:.5f}_pr_{:.5f}_roc_{:.5f}.pth".format(best_f1_epoch, ba, pr_auc, roc_auc)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)

    def train_for_regression(self):
        corrcoef_best = 0
        r2_best = float('-inf')
        rmse_best = 0
        best_r2_epoch = 0
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            loss_sum = torch.zeros((), device=self.device)
            for batch_index, (x, y) in enumerate(tqdm(self.data_loader['train'], mininterval=10)):
                self.optimizer.zero_grad(set_to_none=True)
                x = x.to(self.device, non_blocking=self.use_amp)
                y = y.to(self.device, non_blocking=self.use_amp)
                pred = self.model(x)
                loss = self.criterion(pred, y)

                self.ensure_finite_loss(loss, epoch, batch_index)
                loss.backward()
                loss_sum += loss.detach()
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.update_ema()
                self.optimizer_scheduler.step()

            mean_loss = (loss_sum / self.data_length).item()
            current_lr = self.optimizer.param_groups[0]['lr']

            with torch.no_grad():
                eval_model = self.evaluation_model()
                corrcoef, r2, rmse = self.val_eval.get_metrics_for_regression(eval_model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        mean_loss,
                        corrcoef,
                        r2,
                        rmse,
                        current_lr,
                        (timer() - start_time) / 60
                    )
                )
                if r2 > r2_best:
                    print("r2 increasing....saving weights !! ")
                    print("Val Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format(
                        corrcoef,
                        r2,
                        rmse,
                    ))
                    best_r2_epoch = epoch + 1
                    corrcoef_best = corrcoef
                    r2_best = r2
                    rmse_best = rmse
                    self.best_model_states = copy.deepcopy(eval_model.state_dict())
                    self.save_best_model_state()

        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            corrcoef, r2, rmse = self.test_eval.get_metrics_for_regression(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format(
                    corrcoef,
                    r2,
                    rmse,
                )
            )

            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_corrcoef_{:.5f}_r2_{:.5f}_rmse_{:.5f}.pth".format(best_r2_epoch, corrcoef, r2, rmse)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)

    @staticmethod
    def ensure_finite_loss(loss, epoch, batch_index):
        if not torch.isfinite(loss).all():
            raise FloatingPointError(
                'Non-finite training loss at epoch {}, batch {}: {}'.format(
                    epoch + 1,
                    batch_index + 1,
                    loss.detach().cpu().item(),
                )
            )

    def resolved_selection_metric(self, task):
        metric = self.params.selection_metric
        if metric == 'auto':
            return 'roc_auc' if task == 'binary' else 'kappa'
        valid = {
            'binary': {'ba', 'pr_auc', 'roc_auc'},
            'multiclass': {'ba', 'kappa', 'f1'},
        }
        if metric not in valid[task]:
            raise ValueError(
                'Selection metric {} is not valid for {} task'.format(metric, task)
            )
        return metric

    def binary_selection_score(self, ba, pr_auc, roc_auc):
        return {
            'ba': ba,
            'pr_auc': pr_auc,
            'roc_auc': roc_auc,
        }[self.resolved_selection_metric('binary')]

    def multiclass_selection_score(self, ba, kappa, f1):
        return {
            'ba': ba,
            'kappa': kappa,
            'f1': f1,
        }[self.resolved_selection_metric('multiclass')]
