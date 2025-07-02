import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import h5py
from sklearn.model_selection import train_test_split
import glob

class Patch_Feature_Dataset(Dataset):
    def __len__(self) -> int:
        return len(self.pair_list)

    def __getitem__(self, index):
        img_path, label = self.pair_list[index]
        
        with h5py.File(img_path, 'r') as file:
            patch_feats = file['features'][:]
            patch_feats = torch.tensor(patch_feats)
            coords = file['coords'][:]
        
        coords = torch.tensor(coords)

        return patch_feats, coords, label

    def __init__(self, pair_list, args, train):
        self.args = args
        self.pair_list = pair_list

def prepare_tcga_nsclc(h5_path):
    label_file = 'dataset/csv_files/classification/TCGA-NSCLC-label.csv.zip'
    label_file = pd.read_csv(label_file)

    split_file = f'dataset/csv_files/classification/TCGA-NSCLC-split.csv'
    split_data = pd.read_csv(split_file)

    train_list = []
    val_list = []
    test_list = []

    map = {'LUAD': 0,
           'LUSC': 1}

    for file in os.listdir(h5_path):
            if file[:-3] in split_data['train'].values:
                try:
                    idx = label_file.index[label_file['slide_id'] == file[:-3]+'.svs'].tolist()[0]
                    label = map[label_file.at[idx, 'oncotree_code']]
                except:
                    continue
                train_list.append((os.path.join(h5_path, file), int(label)))
            elif file[:-3] in split_data['val'].values:
                try:
                    idx = label_file.index[label_file['slide_id'] == file[:-3]+'.svs'].tolist()[0]
                    label = map[label_file.at[idx, 'oncotree_code']]
                except:
                    continue
                val_list.append((os.path.join(h5_path, file), int(label)))
            elif file[:-3] in split_data['test'].values:
                try:
                    idx = label_file.index[label_file['slide_id'] == file[:-3]+'.svs'].tolist()[0]
                    label = map[label_file.at[idx, 'oncotree_code']]
                except:
                    continue
                test_list.append((os.path.join(h5_path, file), int(label)))
    
    return train_list, val_list, test_list

def prepare_tcga_brca(h5_path):
    label_file = 'dataset/csv_files/classification/TCGA-BRCA-label.csv'
    label_file = pd.read_csv(label_file)

    split_file = f'dataset/csv_files/classification/TCGA-BRCA-split.csv'
    split_data = pd.read_csv(split_file)

    train_list = []
    val_list = []
    test_list = []

    map = {'IDC': 0,
           'ILC': 1}

    for file in os.listdir(h5_path):
        if file[-2:] == 'h5':
            if file[:-3] in split_data['train'].values:
                try:
                    idx = label_file.index[label_file['slide_id'] == file[:-3]+'.svs'].tolist()[0]
                    label = map[label_file.at[idx, 'oncotree_code']]
                except:
                    continue
                train_list.append((os.path.join(h5_path, file), int(label)))
            elif file[:-3] in split_data['val'].values:
                try:
                    idx = label_file.index[label_file['slide_id'] == file[:-3]+'.svs'].tolist()[0]
                    label = map[label_file.at[idx, 'oncotree_code']]
                except:
                    continue
                val_list.append((os.path.join(h5_path, file), int(label)))
            elif file[:-3] in split_data['test'].values:
                try:
                    idx = label_file.index[label_file['slide_id'] == file[:-3]+'.svs'].tolist()[0]
                    label = map[label_file.at[idx, 'oncotree_code']]
                except:
                    continue
                test_list.append((os.path.join(h5_path, file), int(label)))
    
    return train_list, val_list, test_list

def prepare_dhmc_wsi(h5_path):
    data_csv = 'dataset/csv_files/classification/DHMC.csv'
    data_csv = pd.read_csv(data_csv)

    train_list = []
    val_list = []
    test_list = []

    def mapping_label(label):
        mapping = {
            'Benign': 0,
            'Chromophobe': 1,
            'Clearcell': 2,
            'Papillary': 3,
            'Oncocytoma': 4,
        }
        return mapping[label]

    for file in os.listdir(h5_path):
        path = (os.path.join(h5_path, file))
        idx = data_csv.index[data_csv['File Name'] == (file[:-3])].tolist()[0]
        label = mapping_label(data_csv.at[idx, 'Diagnosis'])
        split = data_csv.at[idx, 'Data Split']
        if split == 'Train':
            train_list.append((path, label)) 
        elif split == 'Val':
            val_list.append((path, label))
        elif split == 'Test':
            test_list.append((path, label))
    return train_list, val_list, test_list

def prepare_panda_wsi(h5_path):
    data_csv = 'dataset/csv_files/classification/PANDA.csv'
    data_csv = pd.read_csv(data_csv)
    data_list = []

    def mapping_label(label):
        return label

    for file in os.listdir(h5_path):
        path = (os.path.join(h5_path, file))
        idx = data_csv.index[data_csv['image_id'] == file[:-3]].tolist()[0]
        label = data_csv.at[idx, 'isup_grade']
        label = mapping_label(label)
        data_list.append((path, label))
    
    x_values = [x for x, _ in data_list]
    y_values = [y for _, y in data_list]

    x_train, x_remaining, y_train, y_remaining = train_test_split(x_values, y_values, 
                                                                  train_size=0.8, 
                                                                  stratify=y_values,
                                                                  random_state=2010
                                                                  )
    x_val, x_test, y_val, y_test = train_test_split(x_remaining, y_remaining, 
                                                    train_size=0.5, stratify=y_remaining,
                                                    random_state=2010)

    train_list = []
    val_list = []
    test_list = []

    for i in range(len(x_train)):
        train_list.append((x_train[i], y_train[i]))
    for i in range(len(x_val)):
        val_list.append((x_val[i], y_val[i]))
    for i in range(len(x_test)):
        test_list.append((x_test[i], y_test[i]))

    return train_list, val_list, test_list

def prepare_bracs_wsi(h5_path):
    train_list = []
    val_list = []
    test_list = []

    label_file = 'dataset/csv_files/classification/BRACS.csv'
    label_file = pd.read_csv(label_file)

    def mapping_label(label):
        mapping = {
                'AT': 0,
                'BT': 1,
                'MT': 2
        }
        return mapping[label]

    for file in os.listdir(h5_path):
        path = (os.path.join(h5_path, file))
        idx = label_file.index[label_file['slide_id'] == file[:-3]].tolist()[0]
        label = label_file.at[idx, 'label']
        label = mapping_label(label)
        split = label_file.at[idx, 'split']

        if split == 'train':
            train_list.append((path, label))
        elif split == 'val':
            val_list.append((path, label))
        elif split == 'test':
            test_list.append((path, label))
    
    return train_list, val_list, test_list

def prepare_data(args):
    if args.task == 'BRACS':
        args.n_classes=3
        args.mamba_2d_max_w = 181272
        args.mamba_2d_max_h = 88334
        return prepare_bracs_wsi(args.h5_path)
    elif args.task == 'BRCA':
        args.n_classes=2
        args.mamba_2d_max_w = 212297
        args.mamba_2d_max_h = 250048
        return prepare_tcga_brca(args.h5_path)
    elif args.task == 'NSCLC':
        args.n_classes=2
        args.mamba_2d_max_w = 197796
        args.mamba_2d_max_h = 110976
        return prepare_tcga_nsclc(args.h5_path)
    elif args.task == 'PANDA':
        args.n_classes=6
        args.mamba_2d_max_w = 96800
        args.mamba_2d_max_h = 47408
        return prepare_panda_wsi(args.h5_path)
    elif args.task == 'DHMC':
        args.n_classes=5
        args.mamba_2d_max_w = 20160
        args.mamba_2d_max_h = 12096
        return prepare_dhmc_wsi(args.h5_path)
    else:
        raise NotImplementedError
