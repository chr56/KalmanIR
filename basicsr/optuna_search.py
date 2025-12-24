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
from basicsr.utils.validation_util import calculate_best_average_metrics, EarlyStoppingWatcher
from basicsr.utils.misc import get_nested_dict_value, set_nested_dict_value


def prepare_cuda():
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    finally:
        time.sleep(1)


class PrunerWatcher(EarlyStoppingWatcher):
    def __init__(
            self,
            trial: optuna.Trial,
            watch_target_metric: str,
            watch_target_dataset: str,
            watch_since: int,
    ):
        super().__init__()
        self.trial = trial
        self.watch_target_metric = watch_target_metric
        self.watch_target_dataset = watch_target_dataset
        self.watch_since = int(watch_since)

    def report(self, step: int, val_dataset_name: str, metric: Dict[str, Any]):
        try:
            if int(step) <= self.watch_since or val_dataset_name != self.watch_target_dataset:
                return  # Skip
            current = metric.get(self.watch_target_metric, None)
            if current is not None and isinstance(current, float):
                self.trial.report(current, step)
        except ValueError as e:
            print(f"Failed to report: {e}")

    def commit(self) -> bool:
        if self.trial.should_prune():
            raise optuna.TrialPruned()
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
        )
    else:
        pruner_watcher = None

    print("================================")
    print(f"Start Optuna Trial {trial.number}")
    print("================================")
    print(f"Settings: \n{trial.params}")
    print("================================")
    sys.stdout.flush()

    try:
        prepare_cuda()

        engine = SupervisedTrainEngine(opt, opt_path=optune_opt_path, dump_real_option=True)
        engine.install_early_stopping_watcher(pruner_watcher)
        model = engine.fit()

        metric_config = optune_opt['metric_to_optimize']
        result_dict = getattr(model, metric_config['result_attribute'])

        if metric_config.get('key_path', None):
            # manual
            final_metric = get_nested_dict_value(result_dict, metric_config['key_path'])
        elif metric_config.get('metric_name', None):
            # auto
            final_metric = calculate_best_average_metrics(result_dict)[metric_config['metric_name']]
        else:
            raise ValueError(f"Unsupported metric: {metric_config}")

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
        return -1.0 if optune_opt['optimization_direction'] == 'maximize' else float('inf')


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
    storage_file = f"sqlite:///{optune_opt['persistent_sqlite_name']}"
    startup_trials = int(optune_opt['num_trials'] * 0.3679)
    study = optuna.create_study(
        study_name=optune_opt['study_name'],
        direction=optune_opt['optimization_direction'],
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=startup_trials,
            n_ei_candidates=int(startup_trials * 1.2),
        ),
        storage=storage_file,
        load_if_exists=args.auto_resume,
    )
    print(f"Starting Optuna study '{study.study_name}':")
    print(f"  - Direction: {study.direction.name}")
    print(f"  - Sampler: {study.sampler.__class__.__name__}")
    print(f"  - Total Trials: {optune_opt['num_trials']} (startup {startup_trials})")
    print(f"  - Storage: {storage_file} (resume: {args.auto_resume})")

    # Start the optimization
    study.optimize(
        lambda trial: objective(
            trial, optune_opt, optuna_opt_path, opt
        ),
        n_trials=optune_opt['num_trials']
    )

    # Print results
    sys.stdout.flush()
    print("\n--- OPTIMIZATION FINISHED ---")
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best metric value: {study.best_value}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
