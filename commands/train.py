import os
import time
import datetime
import copy

import torch
from torch.nn import DataParallel

from commands.test import test
from models.image_classification import ImageClassification
from utils.seed import set_random_seed
from utils.output_manager import OutputManager
from utils.pd_logger import PDLogger


class _StopTraining(Exception):
    pass


def count_params(model):
    count = 0
    count_not_score = 0
    count_reduced = 0
    for (n, p) in model.named_parameters():
        count += p.flatten().size(0)
        print(f'{n}:{p.flatten().size(0)} ({p.size()})')
        count_not_score += p.flatten().size(0)
    count_after_pruning = count_not_score - count_reduced
    total_sparsity = 1 - (count_after_pruning / count_not_score)
    print('Params after/before pruned:\t', count_after_pruning, '/', count_not_score, '(sparsity: ' + str(total_sparsity) + ')')
    print('Total Params:\t', count)
    return {
        'params_after_pruned': count_after_pruning,
        'params_before_pruned': count_not_score,
        'total_params': count,
        'sparsity': total_sparsity,
    }


def train(exp_name, cfg, gpu_id, prefix="", skip_test=False):
    cfg = copy.deepcopy(cfg)

    # Parse list-like YAML fields
    if type(cfg['checkpoint_epochs']) is str:
        cfg['checkpoint_epochs'] = eval(cfg['checkpoint_epochs'])

    checkpoint_iters = cfg.get('checkpoint_iters', None)
    if type(checkpoint_iters) is str:
        checkpoint_iters = eval(checkpoint_iters)
    if checkpoint_iters is not None:
        checkpoint_iters = sorted(set(int(x) for x in checkpoint_iters))
        checkpoint_iters_set = set(checkpoint_iters)
    else:
        checkpoint_iters_set = None

    max_iters = cfg.get('max_iters', None)
    if type(max_iters) is str:
        max_iters = int(eval(max_iters))
    if max_iters is not None:
        max_iters = int(max_iters)

    # Seed
    if cfg['seed'] is not None:
        set_random_seed(cfg['seed'])
    elif cfg['seed_by_time']:
        set_random_seed(int(time.time() * 1000) % 1000000)
    else:
        raise Exception("Set seed value.")

    device = torch.device(f'cuda:{gpu_id}' if cfg['use_cuda'] and torch.cuda.is_available() else 'cpu')
    outman = OutputManager(cfg['output_dir'], exp_name, cfg['output_prefix_hashing'])

    dump_path = outman.get_abspath(prefix=f"dump.{prefix}", ext="pth")
    outman.print('Number of available gpus: ', torch.cuda.device_count(), prefix=prefix)

    pd_logger = PDLogger()
    pd_logger.set_filename(outman.get_abspath(prefix=f"pd_log.{prefix}", ext="pickle"))
    if os.path.exists(pd_logger.filename) and not cfg.get('force_restart', False):
        pd_logger.load()

    if cfg['learning_framework'] == 'ImageClassification':
        learner = ImageClassification(outman, cfg, device, cfg['data_parallel'])
    else:
        raise NotImplementedError

    params_info = count_params(learner.model)

    # Resume / init checkpoint
    start_epoch = 0
    outman.print(dump_path, prefix=prefix)
    try:
        ckp = outman.load_checkpoint(prefix=f"dump.{prefix}", ext="pth")
    except:
        ckp = None

    if (ckp is not None) and (not cfg.get('force_restart', False)):
        start_epoch = ckp.epoch + 1
        if isinstance(learner.model, DataParallel):
            learner.model.module.load_state_dict(ckp.model_state_dict)
        else:
            learner.model.load_state_dict(ckp.model_state_dict)
        learner.optimizer.load_state_dict(ckp.optim_state_dict)
        learner.scheduler.load_state_dict(ckp.sched_state_dict)
    else:
        ckp = outman.new_checkpoint(
            epoch=0,
            total_iters=0,
            best_val=None,
            best_epoch=0,
            total_seconds=0.,
            model_state_dict=None,
            optim_state_dict=None,
            sched_state_dict=None,
        )

        load_checkpoint_flag = (cfg.get('load_checkpoint_by_hparams', None) is not None) or (cfg.get('load_checkpoint_by_path', None) is not None)
        if load_checkpoint_flag:
            if cfg['load_checkpoint_by_hparams'] is not None:
                checkpoint_exp_name = cfg['load_checkpoint_by_hparams']['_exp_name']
                del cfg['load_checkpoint_by_hparams']['_exp_name']
                checkpoint_hparams = cfg['load_checkpoint_by_hparams']
                if cfg.get('seed_checkpoint', None) is not None:
                    checkpoint_hparams['seed'] = cfg['seed_checkpoint']

                checkpoint_outman = OutputManager(
                    cfg['output_dir'],
                    checkpoint_exp_name,
                    cfg['__other_configs__'][checkpoint_exp_name]['output_prefix_hashing']
                )

                checkpoint_job_name = ""
                for k, v in checkpoint_hparams.items():
                    checkpoint_job_name += f"{k}_{v}--"

                model_ckp = checkpoint_outman.load_checkpoint(prefix=f"dump.{checkpoint_job_name}", ext="pth")
            elif cfg['load_checkpoint_by_path'] is not None:
                model_ckp = load_checkpoint_from_path(cfg['load_checkpoint_by_path'])
            else:
                raise NotImplementedError()

            assert model_ckp is not None
            if isinstance(learner.model, DataParallel):
                raise NotImplementedError()
            else:
                if cfg.get('initialize_head', False):
                    learner.model.load_state_dict_without_head(model_ckp.model_state_dict)
                    learner.model.initialize_head(seed=cfg['seed'] * 2 + 1)
                else:
                    learner.model.load_state_dict(model_ckp.model_state_dict)

        # Save init as epoch-1 if requested
        if -1 in cfg['checkpoint_epochs']:
            if isinstance(learner.model, DataParallel):
                ckp.model_state_dict = learner.model.module.state_dict()
            else:
                ckp.model_state_dict = learner.model.state_dict()
            ckp.optim_state_dict = learner.optimizer.state_dict()
            ckp.sched_state_dict = learner.scheduler.state_dict()
            outman.save_checkpoint(ckp, prefix=f'epoch-1.{prefix}', ext="pth")
            outman.save_checkpoint(ckp, prefix=f"dump.{prefix}", ext="pth")

        # Save iter0 if requested
        if checkpoint_iters_set is not None and 0 in checkpoint_iters_set:
            prev_epoch = ckp.epoch
            prev_iters = ckp.total_iters
            ckp.epoch = 0
            ckp.total_iters = 0
            if isinstance(learner.model, DataParallel):
                ckp.model_state_dict = learner.model.module.state_dict()
            else:
                ckp.model_state_dict = learner.model.state_dict()
            ckp.optim_state_dict = learner.optimizer.state_dict()
            ckp.sched_state_dict = learner.scheduler.state_dict()
            outman.save_checkpoint(ckp, prefix=f'iter0.{prefix}', ext="pth")
            ckp.epoch = prev_epoch
            ckp.total_iters = prev_iters

    # A callback that is ALWAYS compatible with ImageClassification.train signature
    def _after_cb(*args, **kwargs):
        # expected: (model, epoch, total_iters, iters_per_epoch)
        if checkpoint_iters_set is None and max_iters is None:
            return

        if len(args) >= 3:
            epoch = args[1]
            total_iters = args[2]
        else:
            # if some caller passes weird signature, just do nothing
            return

        # save iter checkpoints
        if checkpoint_iters_set is not None and int(total_iters) in checkpoint_iters_set:
            prev_epoch = ckp.epoch
            prev_iters = ckp.total_iters

            ckp.epoch = int(epoch)
            ckp.total_iters = int(total_iters)

            if isinstance(learner.model, DataParallel):
                ckp.model_state_dict = learner.model.module.state_dict()
            else:
                ckp.model_state_dict = learner.model.state_dict()
            ckp.optim_state_dict = learner.optimizer.state_dict()
            ckp.sched_state_dict = learner.scheduler.state_dict()

            outman.save_checkpoint(ckp, prefix=f'iter{ckp.total_iters}.{prefix}', ext="pth")

            ckp.epoch = prev_epoch
            ckp.total_iters = prev_iters

        # stop early (CPU budget)
        if max_iters is not None and int(total_iters) >= max_iters:
            raise _StopTraining()

    # Training loop
    stop_training = False
    for _epoch in range(start_epoch, cfg['epoch']):
        ckp.epoch = _epoch
        start_sec = time.time()

        outman.print('[', str(datetime.datetime.now()), '] Epoch: ', str(ckp.epoch), prefix=prefix)

        try:
            # IMPORTANT: we pass an after_callback that accepts the 4-arg call
            results_train = learner.train(ckp.epoch, ckp.total_iters, after_callback=_after_cb)
        except _StopTraining:
            # learner stopped mid-epoch; we still want to log/save a consistent dump below
            outman.print(f"[train] Reached max_iters={max_iters}. Early stop.", prefix=prefix)
            stop_training = True
            # We still need results_train to exist for logging; minimal fallback
            results_train = {
                'moving_accuracy': float('nan'),
                'per_iteration': [],
                'iterations': ckp.total_iters,
                'loss': float('nan'),
            }

        train_accuracy = results_train.get('moving_accuracy', float('nan'))
        results_per_iter = results_train.get('per_iteration', [])
        new_total_iters = results_train.get('iterations', ckp.total_iters)
        total_loss_train = results_train.get('loss', float('nan'))

        pd_logger.add('train_accs', indices=[ckp.epoch], values=[train_accuracy])
        outman.print('Train Accuracy:', str(train_accuracy), prefix=prefix)
        if cfg.get('print_train_loss', False):
            outman.print('Train Loss:', str(total_loss_train), prefix=prefix)

        # Evaluate (optional but keep behaviour identical)
        results_eval = learner.evaluate()
        val_accuracy = results_eval['accuracy']
        pd_logger.add('val_accs', indices=[ckp.epoch], values=[val_accuracy])
        outman.print('Val Accuracy:', str(val_accuracy), prefix=prefix)

        # Save train losses per iteration (if available)
        if len(results_per_iter) > 0:
            losses = [res['mean_loss'] for res in results_per_iter if 'mean_loss' in res]
            if len(losses) > 0:
                start_it = ckp.total_iters
                indices = list(range(start_it, start_it + len(losses)))
                pd_logger.add('train_losses', indices=indices, values=losses)

        # Update total_iters
        ckp.total_iters = int(new_total_iters)

        # Save best model flag
        if (ckp.best_val is None) or (ckp.best_val < val_accuracy):
            ckp.best_val = val_accuracy
            ckp.best_epoch = ckp.epoch
            save_best_model = True
        else:
            save_best_model = False

        end_sec = time.time()
        ckp.total_seconds += end_sec - start_sec

        if isinstance(learner.model, DataParallel):
            ckp.model_state_dict = learner.model.module.state_dict()
        else:
            ckp.model_state_dict = learner.model.state_dict()
        ckp.optim_state_dict = learner.optimizer.state_dict()
        ckp.sched_state_dict = learner.scheduler.state_dict()

        outman.save_checkpoint(ckp, prefix=f"dump.{prefix}", ext="pth")
        if save_best_model and cfg.get('save_best_model', True):
            outman.save_checkpoint(ckp, prefix=f"best.{prefix}", ext="pth")
        if ckp.epoch in cfg['checkpoint_epochs']:
            outman.save_checkpoint(ckp, prefix=f'epoch{ckp.epoch}.{prefix}', ext="pth")

        info_ckp = outman.new_checkpoint(
            last_val=val_accuracy,
            epoch=ckp.epoch,
            best_val=ckp.best_val,
            best_epoch=ckp.best_epoch,
            loss_train=total_loss_train,
            acc_train=train_accuracy,
            total_time=str(datetime.timedelta(seconds=int(ckp.total_seconds))),
            total_seconds=ckp.total_seconds,
            prefix=prefix,
            params_info=params_info,
        )
        outman.save_checkpoint_as_json(info_ckp, prefix=f"info.{prefix}")
        pd_logger.save()

        if stop_training:
            break

    if not skip_test:
        return test(exp_name, cfg, gpu_id, prefix=prefix)
    else:
        return None
