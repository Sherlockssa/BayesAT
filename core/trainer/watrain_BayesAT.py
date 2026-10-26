import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm as tqdm
import core
from core.metrics import accuracy
from core.models import create_model
from core.trainer import ctx_noparamgrad_and_eval, set_bn_momentum, seed, CosineLR
from method.criterion import MILoss
from method.loss_BayesAT import BayesAT_loss


class WATrainer(object):

    def __init__(self, info, args, logger):
        super(WATrainer, self).__init__()
        seed(args.seed)
        self.device = args.device
        self.logger, self.params = logger, args
        self.model = create_model(args=args, info=info, device=self.device, logger=logger)
        self.wa_model = None
        if args.tau > 0:
            self.wa_model = copy.deepcopy(self.model)
        self.eval_attack = core.metrics.CW_PGD(self.get_model(), eps=8 / 255, alpha=1 / 255, steps=20)
        self.init_optimizer(self.params.num_adv_epochs)
        self.criterion_contrastive = None
        self.temperature_schedule = args.temperature_schedule
        if args.method == 'wscat3' or args.method == 'wscat':
            self.criterion_contrastive = MILoss(args).to(self.device)
            param_groups = list(self.criterion_contrastive.parameters())
            if not isinstance(param_groups[0], dict):
                param_groups = [{'params': param_groups}]
            for param_group in param_groups:
                self.optimizer.add_param_group(param_group)
        self.init_scheduler(self.params.num_adv_epochs)
        self.params = args

        # Temperature parameters
        self.temperature_max = getattr(args, 'temperature_max', 3.0)  # Initial temperature
        self.temperature_min = getattr(args, 'temperature_min', 0.5)  # Final temperature
        self.temperature_ramp_up = getattr(args, 'temperature_ramp_up', 30)  # Temperature adjustment period

        if args.pre_resume_path:
            self.pseudo_label_model = create_model(args=args, info=info, device=self.device, logger=None)
            path = args.pre_resume_path
            checkpoint = torch.load(path, weights_only=False)
            if 'model_state_dict' not in checkpoint:
                raise RuntimeError('Model weights not found at {}.'.format(path))
            self.pseudo_label_model.load_state_dict(checkpoint['model_state_dict'])
            pre_epoch = checkpoint['epoch']
            logger.log(f'Pseudo labels by {args.pre_resume_path} (pre-traind for {pre_epoch} epoch)')

    def init_optimizer(self, num_epochs):

        def group_weight(model):
            group_decay = []
            group_no_decay = []
            for n, p in model.named_parameters():
                if 'batchnorm' in n:
                    group_no_decay.append(p)
                else:
                    group_decay.append(p)
            assert len(list(model.parameters())) == len(group_decay) + len(group_no_decay)
            groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
            return groups

        self.optimizer = torch.optim.SGD(group_weight(self.model), lr=self.params.lr,
                                         weight_decay=self.params.weight_decay,
                                         momentum=0.9, nesterov=self.params.nesterov)
        if num_epochs <= 0:
            return

    def init_scheduler(self, num_epochs):
        """
        Initialize scheduler.
        """
        if self.params.scheduler == 'step':
            # scheduler for Wide-Resnet pepar
            milestones_frac, gamma = [0.3, 0.6, 0.8], 0.2
            milestones = []
            for i in milestones_frac:
                milestones.append(int(i * self.params.num_adv_epochs))
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, gamma=gamma, milestones=milestones)
            if self.logger:
                self.logger.log(f'LR scheduler: step milestones-{milestones} gamma-{gamma}')

        elif self.params.scheduler == 'step_trades':
            # scheduler for TRADES
            milestones_frac, gamma = [0.75, 0.9], 0.1
            milestones = []
            for i in milestones_frac:
                milestones.append(int(i * self.params.num_adv_epochs))
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                  gamma=gamma, milestones=milestones)
            if self.logger:
                self.logger.log(f'LR scheduler: step-trades milestones-{milestones} gamma-{gamma}')

        elif self.params.scheduler == 'cosine':
            self.scheduler = CosineLR(self.optimizer, max_lr=self.params.lr, epochs=int(num_epochs))
            if self.logger:
                self.logger.log(f'LR scheduler: cosine max_lr-{self.params.lr} epochs-{int(num_epochs)}')

        elif self.params.scheduler == 'cosinew':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.params.lr, pct_start=0.025,
                                                                 total_steps=int(num_epochs))
            if self.logger:
                self.logger.log(
                    f'LR scheduler: cosinew max_lr-{self.params.lr} pct_start-{0.025} epochs-{int(num_epochs)}')

        elif self.params.scheduler == 'none':
            self.scheduler = None
            if self.logger:
                self.logger.log(f'No scheduler: lr-{self.params.lr}')

        else:
            raise NotImplementedError(self.params.scheduler)

    # Modified _get_current_temperature method from watrain_F1.py
    def _get_current_temperature(self, global_step, total_steps):
        """
        Calculate current temperature value, supporting multiple temperature scheduling methods (including warming up and decay).
        """
        if total_steps <= 0:
            return 1.0

        # Calculate progress (0 to 1)
        progress = min(global_step / (self.temperature_ramp_up * self.update_steps), 1.0)

        # ------------------- Decay version (original) -------------------
        if self.temperature_schedule == 'exponential':
            # Exponential decay: T_max → T_min
            k = getattr(self.params, 'temperature_decay_rate', 3.0)
            current_temp = self.temperature_min + (self.temperature_max - self.temperature_min) * np.exp(-k * progress)

        elif self.temperature_schedule == 'cosine':
            # Cosine decay: T_max → T_min
            current_temp = self.temperature_min + 0.5 * (self.temperature_max - self.temperature_min) * (
                    1 + np.cos(np.pi * progress))

        elif self.temperature_schedule == 'linear':
            # Linear decay: T_max → T_min
            current_temp = self.temperature_max - (self.temperature_max - self.temperature_min) * progress

        # ------------------- Warming up version (newly added) -------------------
        elif self.temperature_schedule == 'exponential_up':
            # Exponential warming up: T_min → T_max
            # Mathematical form: T = T_max - (T_max - T_min) * exp(-k * progress)
            k = getattr(self.params, 'temperature_rise_rate', 3.0)
            current_temp = self.temperature_max - (self.temperature_max - self.temperature_min) * np.exp(-k * progress)

        elif self.temperature_schedule == 'cosine_up':
            # Cosine warming up: T_min → T_max
            # Mathematical form: T = T_min + 0.5 * (T_max - T_min) * (1 - cos(π * progress))
            current_temp = self.temperature_min + 0.5 * (self.temperature_max - self.temperature_min) * (
                    1 - np.cos(np.pi * progress))

        elif self.temperature_schedule == 'linear_up':
            # Linear warming up: T_min → T_max (for reference)
            current_temp = self.temperature_min + (self.temperature_max - self.temperature_min) * progress

        else:
            # Fixed temperature
            current_temp = 1.0

        return current_temp



    def train(self, dataloader, epoch=0, verbose=False, logger=None):

        metrics = pd.DataFrame()
        self.model.train()
        update_iter = 0
        for data in tqdm(dataloader, desc='Epoch {}: '.format(epoch), disable=not verbose):
            update_iter += 1
            self.update_steps = len(dataloader)
            global_step = (epoch - 1) * self.update_steps + update_iter
            self.warmup_steps = 0.025 * self.params.num_adv_epochs * self.update_steps

            # Calculate current temperature
            current_temperature = self._get_current_temperature(global_step,
                                                                self.params.num_adv_epochs * self.update_steps)

            if global_step == 1:
                # make BN running mean and variance init same as Haiku
                set_bn_momentum(self.model, momentum=1.0)
            elif global_step == 2:
                set_bn_momentum(self.model, momentum=0.01)

            x, y = data
            x_l, y_l, x_u = self._data(x, y)
            del x, y
            all_x = []
            all_y_soft = []  # Store soft labels
            all_weights = []  # Store weights

            # Process labeled data
            if x_l is not None and len(x_l) > 0:
                with torch.no_grad():
                    # Use t1 to generate soft labels for labeled data (as supervision signal)
                    t1_logits_l = self.pseudo_label_model(x_l)

                    t1_probs_l_y = torch.softmax(t1_logits_l, dim=1)
                    # Apply temperature scaling
                    t1_logits_l_scaled = t1_logits_l / current_temperature
                    t1_probs_l = torch.softmax(t1_logits_l_scaled, dim=1)

                    # Calculate confidence weights (using maximum probability)
                    max_probs_l, _ = torch.max(t1_probs_l_y, dim=1)
                    weights_l = max_probs_l  # Directly use confidence as weight

                    # Use t1 predictions as soft labels (replace original hard labels)
                    all_x.append(x_l)
                    all_y_soft.append(t1_probs_l)
                    all_weights.append(weights_l)

            # Process unlabeled data
            if x_u is not None and len(x_u) > 0:
                with torch.no_grad():
                    t1_logits_u = self.pseudo_label_model(x_u)
                    t1_probs_u_y = torch.softmax(t1_logits_u, dim=1)
                    # Apply temperature scaling
                    t1_logits_u_scaled = t1_logits_u / current_temperature
                    t1_probs_u = torch.softmax(t1_logits_u_scaled, dim=1)

                    # Calculate confidence weights (using maximum probability)
                    max_probs_u, _ = torch.max(t1_probs_u_y, dim=1)
                    weights_u = max_probs_u  # Directly use confidence as weight

                    # Statistics information
                    all_x.append(x_u)
                    all_y_soft.append(t1_probs_u)
                    all_weights.append(weights_u)

            # If no data, skip this batch
            if len(all_x) == 0:
                continue

            # Merge all data
            x_combined = torch.cat(all_x, dim=0)
            y_combined_soft = torch.cat(all_y_soft, dim=0)  # Soft labels
            weights_combined = torch.cat(all_weights, dim=0)  # Weights

            con_ramp = sigmoid_rampup(global_step, 1, self.params.consistency_ramp_up * self.update_steps + 1)
            loss_dict, batch_metrics = BayesAT_loss(criterion_mi=self.criterion_contrastive,
                                                  model=self.model, wamodel=self.wa_model,
                                                  consistency_cost=self.params.consistency_cost * con_ramp,
                                                  x_l=x_combined, y=y_combined_soft, x_u=None, optimizer=self.optimizer,
                                                  step_size=self.params.attack_step,
                                                  epsilon=self.params.attack_eps, perturb_steps=self.params.attack_iter,
                                                  lamb=self.params.lamb, beta=self.params.beta,
                                                  attack=self.params.attack,
                                                  contrast_label=self.params.contrast_label, use_soft_labels=True,
                                                  sample_weights=weights_combined,
                                                  temperature=current_temperature)  # Pass current temperature

            loss_dict['loss'].backward()
            if self.params.clip_grad:
                nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_grad)
            self.optimizer.step()

            ema_update(self.wa_model, self.model, global_step,
                       decay_rate=self.params.tau if epoch <= self.params.consistency_ramp_up else self.params.tau_after,
                       warmup_steps=self.warmup_steps, dynamic_decay=True)
            try:
                metrics = metrics.append(pd.DataFrame(batch_metrics, index=[0]), ignore_index=True)
            except:
                metrics = metrics._append(pd.DataFrame(batch_metrics, index=[0]), ignore_index=True)
            if logger is not None:
                for key in loss_dict:
                    logger.add("training", key, loss_dict[key].item(), global_step)
                # Log current temperature
                logger.add("training", "temperature", current_temperature, global_step)

        if self.params.scheduler not in ['cyclic', 'none']:
            self.scheduler.step()
        if logger is not None:
            logger.log_stats(global_step, ["training"])
        update_bn(self.wa_model, self.model)
        return dict(metrics.mean())

    def _data(self, x, y):
        device = self.device
        idx = np.array(range(len(y)))
        idx_l, idx_u = idx[y != -1], idx[y == -1]
        if len(idx_u) > 0:
            x_u = x[idx_u]
            x_l, y_l = x[idx_l], y[idx_l]
            return x_l.to(device), y_l.to(device), x_u.to(device)
        else:
            return x.to(device), y.to(device), None

    def get_model(self):
        if self.wa_model is None:
            return self.model
        else:
            return self.wa_model

    def eval(self, dataloader, model=None, adversarial=False, verbose=True):
        model = model or self.get_model()
        model.eval()
        acc, total = 0.0, 0
        for data in tqdm(dataloader, desc='Eval : ', disable=not verbose):
            x, y = data
            x, y = x.to(self.device), y.to(self.device)
            total += x.size(0)
            if adversarial:
                with ctx_noparamgrad_and_eval(self.wa_model):
                    x_adv = self.eval_attack(x, y)
                with torch.no_grad():
                    out = model(x_adv)
            else:
                with torch.no_grad():
                    out = model(x)
            preds = out.argmax(dim=1)
            acc += (preds == y).sum().item()
        model.train()
        return acc / total

    def save_model(self, path, epoch):
        save_dict = {
            'model_state_dict': self.wa_model.state_dict(),
            'unaveraged_model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': epoch, }
        if self.params.method == 'wscat3' or self.params.method == 'wscat':
            save_dict['mi_embed1'] = self.criterion_contrastive.embed1.state_dict()
            save_dict['mi_embed2'] = self.criterion_contrastive.embed2.state_dict()
        torch.save(save_dict, path)

    def load_model(self, path, weights_only=False):
        checkpoint = torch.load(path, weights_only=False)
        if 'model_state_dict' not in checkpoint:
            raise RuntimeError('Model weights not found at {}.'.format(path))
        else:
            self.wa_model.load_state_dict(checkpoint['model_state_dict'])
            self.model.load_state_dict(checkpoint['unaveraged_model_state_dict'])
        if not weights_only:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if self.params.method == 'wscat3' or self.params.method == 'wscat':
                self.criterion_contrastive.embed1.load_state_dict(checkpoint['mi_embed1'])
                self.criterion_contrastive.embed2.load_state_dict(checkpoint['mi_embed2'])
        return checkpoint['epoch']


def ema_update(wa_model, model, global_step, decay_rate=0.995, warmup_steps=0, dynamic_decay=True):
    """
    Exponential model weight averaging update.
    """
    factor = int(global_step >= warmup_steps)
    if dynamic_decay:
        delta = global_step - warmup_steps
        decay = min(decay_rate, (1. + delta) / (10. + delta)) if 10. + delta != 0 else decay_rate
    else:
        decay = decay_rate
    decay *= factor

    for p_swa, p_model in zip(wa_model.parameters(), model.parameters()):
        p_swa.data *= decay
        p_swa.data += p_model.data * (1 - decay)


@torch.no_grad()
def update_bn(avg_model, model):
    """
    Update batch normalization layers.
    """
    avg_model.eval()
    model.eval()
    for module1, module2 in zip(avg_model.modules(), model.modules()):
        if isinstance(module1, torch.nn.modules.batchnorm._BatchNorm):
            module1.running_mean = module2.running_mean
            module1.running_var = module2.running_var
            module1.num_batches_tracked = module2.num_batches_tracked


def sigmoid_rampup(global_step, start_iter, end_iter):
    if global_step < start_iter:
        return 0.
    elif start_iter >= end_iter:
        return 1.
    else:
        rampup_length = end_iter - start_iter
        cur_ramp = global_step - start_iter
        cur_ramp = np.clip(cur_ramp, 0, rampup_length)
        phase = 1.0 - cur_ramp / rampup_length
        return np.exp(-5.0 * phase * phase)

