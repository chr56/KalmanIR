import datetime
import logging
import time
from os import path as osp
from typing import Any, Dict

import math
import torch

from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.utils.validation_util import EarlyStoppingWatcher, calculate_best_average_metrics
from basicsr.models import build_model
from basicsr.utils import (
    AvgTimer, MessageLogger, check_resume,
    get_env_info, get_root_logger, get_time_str,
    make_exp_dirs, setup_external_logger, scandir,
    calculate_eta
)
from basicsr.utils.options import (
    copy_opt_file, dump_opt_file, dict2str, parse_val_profiles
)


class SupervisedTrainEngine:
    """
    Pipeline for supervised training.
    """

    def __init__(
            self,
            opt: dict,
            opt_path: str,
            dump_real_option: bool = False,
            print_env_info: bool = True,
            print_config: bool = True,
    ):
        self.opt = opt
        self.opt_path = opt_path
        self.dump_real_option = dump_real_option
        self.print_env_info = print_env_info
        self.print_config = print_config

        self.logger = None
        self.tb_logger = None
        self.msg_logger = None

        self.model = None
        self.train_loader = None
        self.train_sampler = None
        self.val_loaders = None
        self.prefetcher = None

        self.val_profiles = None

        self.early_stopping_watcher = None
        self.flag_early_stopping: bool = False

        self.start_epoch = 0
        self.current_iter = 0
        self.total_epochs = 0
        self.total_iters = 0

    def _check_files(self) -> Any | None:
        """Read resume state and setup dirs and logger"""
        # load resume states if necessary
        resume_state = self.load_resume_state()

        # mkdir for experiments and logger
        if resume_state is None and self.opt['rank'] == 0:
            make_exp_dirs(self.opt)

        # export options by copy
        if self.opt_path is not None:
            copy_opt_file(self.opt_path, self.opt['path']['experiments_root'])
        # export options by dumping
        if self.dump_real_option:
            path = osp.join(self.opt['path']['experiments_root'], f"opt_{self.opt['name']}.yml")
            dump_opt_file(self.opt, path)

        # setup logger
        self.logger = self.install_logger()
        if self.print_config:
            self.logger.info(dict2str(self.opt))
        self.tb_logger = setup_external_logger(self.opt)
        self.msg_logger = MessageLogger(self.opt, self.current_iter, self.tb_logger)
        return resume_state

    def install_logger(self, log_file=None):
        log_file = log_file or osp.join(self.opt['path']['log'], f"train_{self.opt['name']}_{get_time_str()}.log")
        logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
        if self.print_env_info:
            logger.info(get_env_info())
        return logger

    def install_visualizer(self):
        from basicsr.utils.visualizer import Visualizer
        Visualizer.setup(Visualizer(runs=self.opt['name']))

    def install_early_stopping_watcher(self, watcher: EarlyStoppingWatcher):
        self.early_stopping_watcher = watcher

    def load_resume_state(self):
        resume_state_path = None
        if self.opt['auto_resume']:
            state_path = osp.join('experiments', self.opt['name'], 'training_states')
            if osp.isdir(state_path):
                states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
                if len(states) != 0:
                    states = [float(v.split('.state')[0]) for v in states]
                    resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                    self.opt['path']['resume_state'] = resume_state_path
        elif self.opt['path'].get('resume_state'):
            resume_state_path = self.opt['path']['resume_state']

        if resume_state_path is None:
            return None

        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
        check_resume(self.opt, resume_state['iter'])
        return resume_state

    def _train_epoch(self, epoch: int, data_timer: AvgTimer, iter_timer: AvgTimer, start_time: float):
        """Trains the model for one epoch."""
        self.train_sampler.set_epoch(epoch)
        self.prefetcher.reset()
        train_data = self.prefetcher.next()

        while train_data is not None:
            data_timer.record()
            self.current_iter += 1
            if self.current_iter > self.total_iters:
                break

            # update learning rate
            self.model.update_learning_rate(self.current_iter, warmup_iter=self.opt['train'].get('warmup_iter', -1))

            # training
            self.model.feed_data(train_data)
            self.model.optimize_parameters(self.current_iter)

            iter_timer.record()

            if self.current_iter == 1 and self.start_epoch == 0:
                # reset start time in msg_logger for more accurate eta_time
                # not work in resume mode
                self.msg_logger.reset_start_time()

            # log current
            if self.current_iter % self.opt['logger']['print_freq'] == 0:
                self._log_training_state(epoch, iter_timer, data_timer)

            # save models and training states
            if self.current_iter % self.opt['logger']['save_checkpoint_freq'] == 0:
                self.logger.info('Saving models and training states.')
                self.model.save(epoch, self.current_iter)

            # validate
            self._validate(start_time)
            if self.flag_early_stopping: return

            data_timer.start()
            iter_timer.start()
            train_data = self.prefetcher.next()

    def _log_training_state(self, epoch: int, iter_timer: AvgTimer, data_timer: AvgTimer):
        """Logs the current training state."""
        lrs = self.model.get_current_learning_rate()
        log_vars = {
            'epoch': epoch,
            'iter': self.current_iter,
            'lrs': lrs,
            'time': iter_timer.get_avg_time(),
            'data_time': data_timer.get_avg_time(),
        }
        log_vars.update(self.model.get_current_log())
        self.msg_logger(log_vars)
        if self.tb_logger:
            for i, lr in enumerate(lrs):
                self.tb_logger.add_scalar(f'lr_{i}', lr, self.current_iter)

    def _validate(self, training_start_time: float):
        if self.opt.get('val') is None or not self.val_profiles:
            return

        for profile_name, profile in self.val_profiles.items():
            if profile.check(self.current_iter):
                self.msg_logger.logger.info(f"Validating ({profile_name})...")
                validation_time = time.time()
                for val_loader in profile.filter_datasets(self.val_loaders):
                    dataset_name = val_loader.dataset.opt['name']
                    self.model.validation(val_loader, self.current_iter, self.tb_logger, self.opt['val']['save_img'])
                    if self.early_stopping_watcher is not None and hasattr(self.model, "current_metric_results"):
                        self.early_stopping_watcher.report(
                            self.current_iter, dataset_name, self.model.current_metric_results
                        )

                validation_time = time.time() - validation_time

                eta = calculate_eta(self.current_iter, self.total_iters, training_start_time)
                self.msg_logger.logger.info(
                    f"\n\t> Validation took {validation_time:.1f}s. \n\t> ETA: {eta} \n"
                )
                if self.early_stopping_watcher is not None:
                    if self.early_stopping_watcher.commit():
                        self.flag_early_stopping = True
                        return

    def _final_validate(self):
        if self.opt.get('val') and self.val_profiles:
            target_loaders = self.val_loaders
            end_profile = self.val_profiles.get('end')
            if end_profile:
                target_loaders = end_profile.filter_datasets(self.val_loaders)
            for val_loader in target_loaders:
                self.model.validation(val_loader, self.current_iter, self.tb_logger, self.opt['val']['save_img'])

    def get_current_best_result_objective(
            self, target_attribute: str = 'best_metric_results',
            dataset_weights: Dict[str, float] = None,
    ) -> dict:
        if not hasattr(self.model, target_attribute):
            self.logger.error(f"Target attribute {target_attribute} not found!")
            return {}
        best_metric_results: dict = getattr(self.model, target_attribute)
        if not isinstance(best_metric_results, dict):
            self.logger.error(f"Target attribute {target_attribute} is not a dictionary!")
            return {}
        if dataset_weights is None:
            return calculate_best_average_metrics(best_metric_results)

        metric_sums: Dict[str, float] = {}
        for dataset_name, metrics in best_metric_results.items():
            if dataset_name not in dataset_weights: continue
            for metric_name, data in metrics.items():
                if metric_name not in metric_sums:
                    metric_sums[metric_name] = 0.0
                value = float(data.get('val', 0.0))
                metric_sums[metric_name] += value * dataset_weights[dataset_name]

        return metric_sums

    def _setup_prefetcher(self):
        """Initializes the data prefetcher."""
        prefetch_mode = self.opt['datasets']['train'].get('prefetch_mode')
        if prefetch_mode is None or prefetch_mode == 'cpu':
            self.prefetcher = CPUPrefetcher(self.train_loader)
        elif prefetch_mode == 'cuda':
            self.prefetcher = CUDAPrefetcher(self.train_loader, self.opt)
            self.logger.info(f'Use {prefetch_mode} prefetch dataloader')
            if not self.opt['datasets']['train'].get('pin_memory'):
                raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
        else:
            raise ValueError(f"Wrong prefetch_mode {prefetch_mode}. Supported: None, 'cuda', 'cpu'.")

    def _prepare(self):
        """Initializes loggers, dataloaders, model, and training states."""

        # Models & State
        resume_state = self._check_files()
        self.model = build_model(self.opt)
        if resume_state:
            self.model.resume_training(resume_state)
            self.logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
            self.start_epoch = resume_state['epoch']
            self.current_iter = resume_state['iter']

        # Data Loader
        (self.train_loader, self.train_sampler, self.val_loaders,
         self.total_epochs, self.total_iters) = create_train_val_dataloader(self.opt, self.logger)
        self._setup_prefetcher()

        # Val Profiles
        self.val_profiles = parse_val_profiles(self.opt.get('val'), debug=('debug' in self.opt['name']))

    def fit(self):

        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True

        self._prepare()

        self.logger.info(f'Start training from epoch: {self.start_epoch}, iter: {self.current_iter}')

        data_timer, iter_timer = AvgTimer(), AvgTimer()
        start_time = time.time()

        for epoch in range(self.start_epoch, self.total_epochs + 1):
            self._train_epoch(epoch, data_timer, iter_timer, start_time)
            if self.flag_early_stopping: break

        consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
        if self.flag_early_stopping:
            self.logger.error(
                f"Stopping training at {self.current_iter} due to early stopping! Time consumed: {consumed_time}")
        else:
            self.logger.info(f'End of training. Time consumed: {consumed_time}')
            self.logger.info('Saving the latest model.')
            self.model.save(epoch=-1, current_iter=-1)  # -1 for latest
            self._final_validate()

        if self.tb_logger:
            self.tb_logger.close()

        return self.model


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loaders = None, []
    train_sampler = None
    total_iters, total_epochs = 0, 0
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / num_iter_per_epoch)
            logger.info('Training statistics:'
                        f'\n\tNumber of train images: {len(train_set)}'
                        f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                        f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                        f'\n\tWorld size (gpu number): {opt["world_size"]}'
                        f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
        elif phase.split('_')[0] == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_loaders.append(val_loader)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters
