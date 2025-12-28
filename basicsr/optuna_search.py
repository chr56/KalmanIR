import argparse
import copy
import sys
import time
from os import path as osp
from typing import Dict, Any, Literal

import optuna
import yaml

ROOT_PATH = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
sys.path.append(ROOT_PATH)  # Add the parent directory to sys.path for `basicsr`

from basicsr.trainer import SupervisedTrainEngine
from basicsr.utils.options import setup_env_and_update_options, ordered_yaml, populate_paths
from basicsr.utils.validation_util import EarlyStoppingWatcher
from basicsr.utils.misc import set_nested_dict_value


def prepare_cuda():
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    finally:
        time.sleep(1)


def print_dict_items(data: dict):
    for key, value in data.items():
        print(f"\t\t{key}:\t\t{value}")


def invalid_data(direction_maximize: bool = True):
    return -1.0 if direction_maximize else float('inf')


class PrunerWatcher(EarlyStoppingWatcher):
    def __init__(
            self,
            trial: optuna.Trial,
            watch_target_metric: str,
            watch_target_dataset: str,
            watch_since: int,
            patience: int,
    ):
        super().__init__()
        self.trial = trial
        self.watch_target_metric = watch_target_metric
        self.watch_target_dataset = watch_target_dataset
        self.watch_since = int(watch_since)

        self.patience = patience
        self.current_step = 0

    def report(self, step: int, val_dataset_name: str, metric: Dict[str, Any]):
        self.current_step = int(step)
        if self.watch_target_dataset == val_dataset_name:
            try:
                current = metric.get(self.watch_target_metric, None)
                if current is not None and isinstance(current, float):
                    self.trial.report(current, int(step))
            except ValueError as e:
                print(f"Failed to report: {e}")

    def commit(self) -> bool:
        if int(self.current_step) >= self.watch_since:
            sys.stdout.flush()
            if self.trial.should_prune():
                self.patience -= 1
                if self.patience < 0:
                    print(f"Pruning trial at {self.current_step} without patience!")
                    raise optuna.TrialPruned()
                else:
                    print(f"Continue trial with patience!")
            else:
                print("Continue trial!")
        return False


def objective(trial: optuna.Trial, optune_opt: dict, optune_opt_path: str, opt: dict):
    opt = copy.deepcopy(opt)

    # Update Name
    opt['name'] = f"{optune_opt['study_name']}_trial_{trial.number}"

    # Repopulate paths (name updated)
    populate_paths(opt, ROOT_PATH, is_train=True)

    # Modify hyper parameter
    for param in optune_opt['search_space']:
        param_path = param['param_path']
        param_type = param['type']
        param_args = param['args']

        param_name = param.get('name', param_path[-2] + '.' + param_path[-1])

        if param_type == 'float':
            value = trial.suggest_float(param_name, **param_args)
        elif param_type == 'int':
            value = trial.suggest_int(param_name, **param_args)
        elif param_type == 'categorical':
            value = trial.suggest_categorical(param_name, **param_args)
        else:
            raise ValueError(f"Unsupported parameter type: {param_type}")

        set_nested_dict_value(opt, param_path, value)

    # Pruner settings
    pruner_opt = optune_opt['pruner']
    if pruner_opt is not None:
        pruner_watcher = PrunerWatcher(
            trial,
            watch_target_metric=pruner_opt['watch_target_metric'],
            watch_target_dataset=pruner_opt['watch_target_dataset'],
            watch_since=min(pruner_opt['watch_since'], opt['train']['total_iter'] // 2),
            patience=pruner_opt.get('patience', 4),
        )
    else:
        pruner_watcher = None

    print("================================")
    print(f"Start Optuna Trial {trial.number}")
    print("================================")
    print(f"Settings:")
    print_dict_items(trial.params)
    print("================================")
    sys.stdout.flush()

    try:
        prepare_cuda()

        engine = SupervisedTrainEngine(
            opt, opt_path=optune_opt_path, dump_real_option=True, print_env_info=False,
        )
        engine.install_early_stopping_watcher(pruner_watcher)
        engine.fit()

        metric_config = optune_opt['metric_to_optimize']
        final_metric = engine.get_current_best_result_objective(
            target_attribute=metric_config['result_attribute'],
            dataset_weights=metric_config['dataset_weights'],
        ).get(metric_config['metric_name'], invalid_data(optune_opt['optimization_direction'] == 'maximize'))

        sys.stdout.flush()
        print("================================")
        print(f"Trial {trial.number} Finished")
        print("================================")
        print(f"\t\t Metric Score: {final_metric}")
        print("================================")

        return final_metric

    except Exception as e:
        if isinstance(e, optuna.TrialPruned):
            raise optuna.TrialPruned() from e
        print("++++++++++++++++++++++++++++++++")
        print(f" Trial {trial.number} FAILED ---\n{e}")
        print("++++++++++++++++++++++++++++++++")
        time.sleep(1)
        return invalid_data(optune_opt['optimization_direction'] == 'maximize')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to the Optuna YAML configuration file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher')
    parser.add_argument(
        '--auto_resume',
        action='store_true')
    parser.add_argument(
        '--debug',
        action='store_true')
    parser.add_argument(
        '--local-rank',
        type=int, default=0)  # for pytorch 2.0
    args = parser.parse_args()

    optuna_opt_path = args.opt
    with open(optuna_opt_path, 'r') as f:
        optune_opt = yaml.load(f, Loader=ordered_yaml()[0])

    with open(optune_opt['base_config_path'], mode='r') as f:
        # parse yml to dict
        base_opt = yaml.load(f, Loader=ordered_yaml()[0])

    opt = setup_env_and_update_options(
        opt=base_opt,
        launcher=args.launcher,
        auto_resume=args.auto_resume,
        debug=args.debug,
        force_yml=list(),
        root_path=ROOT_PATH,
        is_train=True,
    )

    # Optuna study
    num_trials = int(optune_opt['num_trials'])
    startup_trials = int(optune_opt.get("startup_trials", None) or (num_trials * 0.3679))
    storage_file = f"sqlite:///{optune_opt['persistent_sqlite_name']}"
    study = optuna.create_study(
        study_name=optune_opt['study_name'],
        direction=optune_opt['optimization_direction'],
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=startup_trials,
            n_ei_candidates=int(startup_trials * 1.2),
        ),
        pruner=optuna.pruners.PercentilePruner(
            percentile=25,
            n_startup_trials=min(4, int(startup_trials * 0.8)),
            n_min_trials=4,
        ),
        storage=storage_file,
        load_if_exists=args.auto_resume,
    )
    print(f"Starting Optuna study '{study.study_name}':")
    print(f"  - Direction: {study.direction.name}")
    print(f"  - Sampler: {study.sampler.__class__.__name__}")
    print(f"  - Pruner: {study.pruner.__class__.__name__}")
    print(f"  - Total Trials: {num_trials} (startup {startup_trials})")
    print(f"  - Storage: {storage_file} (resume: {args.auto_resume})")

    # Start the optimization
    study.optimize(
        lambda trial: objective(
            trial, optune_opt, optuna_opt_path, opt
        ),
        n_trials=num_trials
    )

    # Print results
    sys.stdout.flush()
    print("\n--- OPTIMIZATION FINISHED ---")
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best metric value: {study.best_value}")
    print("Best hyperparameters:")
    print_dict_items(study.best_params)


if __name__ == '__main__':
    main()
