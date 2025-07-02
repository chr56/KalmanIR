from __future__ import print_function
import argparse
import os
from utils.utils import *
from utils.core_utils import train, train_survival
from dataset.dataset_generic import *
from dataset.dataset_survival import *
import torch
import numpy as np
import wandb


def main():
    # Generic training settings
    parser = argparse.ArgumentParser(description='Configurations for WSI Training')
    parser.add_argument('--data_root_dir', type=str, default=None, 
                        help='data directory')
    parser.add_argument('--max_epochs', type=int, default=20,
                        help='maximum number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--label_frac', type=float, default=1.0,
                        help='fraction of training labels (default: 1.0)')
    parser.add_argument('--reg', type=float, default=1e-5,
                        help='weight decay (default: 1e-5)')
    parser.add_argument('--seed', type=int, default=1, 
                        help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--k', type=int, default=1, help='number of folds (default: 10)')
    parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
    parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
    parser.add_argument('--results_dir', default='results', help='results directory (default: ./results)')
    parser.add_argument('--split_dir', type=str, default='./splits/BRACS_100', 
                        help='manually specify the set of splits to use, ' 
                        +'instead of infering from the task and label_frac argument (default: None)')
    parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
    parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
    parser.add_argument('--opt', type=str, choices = ['adam', 'sgd', 'adamw'], default='adamw')
    parser.add_argument('--drop_out', type=float, default=0.25, help='enable dropout (p=0.25)')
    parser.add_argument('--model_type', type=str, default='MambaMIL_2D', 
                        help='type of model (default: clam_sb, clam w/ single attention branch)')
    parser.add_argument('--exp_code', type=str, default='demo', help='experiment code for saving results')
    parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
    parser.add_argument('--task', type=str, default='BRACS3')
    parser.add_argument('--h5_path', required=True, type=str)
    parser.add_argument('--patch_size', type=str, default='')
    parser.add_argument('--preloading', type=str, default='no')
    parser.add_argument('--in_dim', type=int, default=1024)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--patch_encoder_batch_size', type=int, default=128)


    ## survival pred
    parser.add_argument('--survival', action='store_true', default=False)
    parser.add_argument('--bag_loss', type=str, default='nll_surv')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--mode', type = str, choices=['path', 'omic', 'pathomic', 'cluster'], default='path', help='which modalities to use')


    ## mambamil
    parser.add_argument('--reverse_ord', default=False, action='store_true')
    parser.add_argument('--pos_emb_type', default=None, choices= [None, 'linear'])
    parser.add_argument('--pos_emb_dropout', type=float, default=0.0)
    parser.add_argument('--input_type',type=str, default='feature_uni')
    parser.add_argument('--train_patch_encoder',action='store_true', default=False)
    parser.add_argument('--shuffle_patch', default=False, action='store_true')
    parser.add_argument('--patch_encoder',type=str, default='feature_uni')

    # mamba config
    parser.add_argument('--mambamil_dim',type=int, default=128)
    parser.add_argument('--mambamil_rate',type=int, default=10) 
    parser.add_argument('--mambamil_state_dim',type=int, default=16)
    parser.add_argument('--mambamil_layer',type=int, default=1)
    parser.add_argument('--mambamil_inner_layernorms',default=False, action='store_true')
    parser.add_argument('--mambamil_type',type=str, default=None, 
                        choices= ['Mamba', 'SRMamba','SimpleMamba'], help='mambamil_type')
    

    parser.add_argument('--pscan',default=True)
    parser.add_argument('--cuda_pscan',default=False, action='store_true')
    parser.add_argument('--mamba_2d',default=False, action='store_true')
    parser.add_argument('--mamba_2d_pad_token', '-p', type=str, default='trainable', 
                        choices= ['zero', 'trainable'])
    parser.add_argument('--mamba_2d_patch_size',type=int, default=512)
    
    args = parser.parse_args()

    wandb.init(project='MambaMIL-2D')
    wandb.config.update(args)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def seed_torch(seed):
        import random
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False

    seed_torch(args.seed)

    settings = {'num_splits': args.k, 
                'k_start': args.k_start,
                'k_end': args.k_end,
                'task': args.task,
                'max_epochs': args.max_epochs, 
                'results_dir': args.results_dir, 
                'lr': args.lr,
                'experiment': args.exp_code,
                'reg': args.reg,
                'label_frac': args.label_frac,
                'seed': args.seed,
                'model_type': args.model_type,
                "use_drop_out": args.drop_out,
                'weighted_sample': args.weighted_sample,
                'opt': args.opt}

    print('\nLoad Dataset')

    if args.survival:
        split_path = f'dataset/csv_files/survival/TCGA_{args.task}_survival_kfold/splits_{args.fold}.csv'
        dataset = Generic_MIL_Survival_Dataset(csv_path = f'dataset/csv_files/survival/{args.task}.csv',
                                            mode = 'path',
                                            apply_sig = False,
                                            data_dir= args.h5_path, 
                                            shuffle = False, 
                                            seed = args.seed, 
                                            print_info = True,
                                            patient_strat= False,
                                            n_bins=4,
                                            label_col = 'survival_months',
                                            ignore=[])
        post_init_survival_size(args)
    else:
        train_set, valid_set, test_set = prepare_data(args=args)
        train_dataset = Patch_Feature_Dataset(
                pair_list=train_set,
                args=args,
                train=True
            )
        val_dataset = Patch_Feature_Dataset(
            pair_list=valid_set,
            args=args,
            train=False
        )
        test_dataset = Patch_Feature_Dataset(
            pair_list=test_set,
            args=args,
            train=False
        )

    
    args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)

    if args.split_dir is None:
        args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))

    settings.update({'split_dir': args.split_dir})

    with open(args.results_dir + '/experiment.txt', 'w') as f:
        print(settings, file=f)
   
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.survival:
        train_dataset, val_dataset, test_dataset = dataset.return_splits(backbone='uni',from_id=False, csv_path=split_path)
        datasets = (train_dataset, val_dataset, test_dataset)
        train_survival(datasets, args)
    else:
        datasets = (train_dataset, val_dataset, test_dataset)
        train(datasets, args)


if __name__ == '__main__':
    wandb.login() 
    main()
