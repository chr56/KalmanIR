import datetime
import sys
import logging
import math
import time
import torch
from os import path as osp

ROOT_PATH = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
sys.path.append(ROOT_PATH)  # Add the parent directory to sys.path for `basicsr`

from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (AvgTimer, MessageLogger, check_resume,
                           get_env_info, get_root_logger, get_time_str,
                           make_exp_dirs, setup_external_logger, scandir)
from basicsr.utils.options import (
    copy_opt_file, dump_opt_file, dict2str, parse_argument_for_options, parse_val_profiles
)


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loaders = None, []
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
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
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


def load_resume_state(opt):
    resume_state_path = None
    if opt['auto_resume']:
        state_path = osp.join('experiments', opt['name'], 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt['path']['resume_state'] = resume_state_path
    else:
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
        check_resume(opt, resume_state['iter'])
    return resume_state


def dump_options(opt: dict, opt_path: str, dump_real_option: bool):
    # copy the yml file to the experiment root
    if opt_path is not None:
        copy_opt_file(opt_path, opt['path']['experiments_root'])
    if dump_real_option:
        path = osp.join(opt['path']['experiments_root'], f"opt_{opt['name']}.yml")
        dump_opt_file(opt, path)


def initialize_logger(log_file):
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    return logger


def train_from_option_file(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, opt_path = parse_argument_for_options(root_path, is_train=True)
    train_pipeline(opt, opt_path=opt_path)


def train_pipeline(opt: dict, opt_path: str, dump_real_option: bool = False):
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # load resume states if necessary
    resume_state = load_resume_state(opt)
    # mkdir for experiments and logger
    if resume_state is None and opt['rank'] == 0:
        make_exp_dirs(opt)

    dump_options(opt, opt_path, dump_real_option)

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    logger = initialize_logger(
        log_file=osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    )
    logger.info(dict2str(opt))
    # initialize wandb and tb loggers
    tb_logger = setup_external_logger(opt)

    # create train and validation dataloaders
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = create_train_val_dataloader(opt, logger)

    # create model
    model = build_model(opt)

    if resume_state:  # resume training
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f"Wrong prefetch_mode {prefetch_mode}. Supported ones are: None, 'cuda', 'cpu'.")

    # validation settings
    val_profiles = parse_val_profiles(opt.get('val'), debug=('debug' in opt['name']))

    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    # stage_timer = AvgTimer(window=4)
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_timer.record()

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)

            iter_timer.record()
            if current_iter == 1:
                # reset start time in msg_logger for more accurate eta_time
                # not work in resume mode
                msg_logger.reset_start_time()
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                lrs = model.get_current_learning_rate()
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': lrs})
                log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)
                for g, lr in enumerate(lrs):
                    if tb_logger is not None: tb_logger.add_scalar(f'lr_{g}', lr, current_iter)

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            # validation
            if opt.get('val') is not None and val_profiles:
                for profile_name, profile in val_profiles.items():
                    if profile.check(current_iter):
                        msg_logger.logger.info(f"Validating ({profile_name})...")

                        validation_time = time.time()
                        for val_loader in profile.filter_datasets(val_loaders):
                            model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
                        validation_time = time.time() - validation_time
                        # stage_timer.record()

                        msg_logger.logger.info(
                            "\n"
                            f"\t> Current validation took {validation_time:.1f}s. \n"
                            # f"\t> Current mini epoch took {stage_timer.get_current_time():.1f}s. \n"
                            f"\t> ETA: {calculate_eta(current_iter, total_iters, start_time)} \n"
                        )
                        pass

            data_timer.start()
            iter_timer.start()
            train_data = prefetcher.next()
        # end of iter

    # end of epoch

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None and val_profiles:
        target_loaders = val_loaders
        end_profile = val_profiles.get('end')
        if end_profile is not None:
            target_loaders = end_profile.filter_datasets(val_loaders)
        for val_loader in target_loaders:
            model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()

    return model


def calculate_eta(current, total, start_time):
    current_time = time.time()
    elapsed_duration = current_time - start_time
    estimated_total_duration = elapsed_duration * (total / current)
    eta = str(datetime.timedelta(seconds=int(estimated_total_duration - elapsed_duration)))
    return eta


if __name__ == '__main__':
    train_from_option_file(ROOT_PATH)
