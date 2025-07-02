import numpy as np
import torch
from utils.utils import *
import os
import torch.nn as nn
from sksurv.metrics import concordance_index_censored
from tqdm import tqdm
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.MambaMIL_2D import MambaMIL_2D
from utils.survival_utils import NLLSurvLoss, CoxSurvLoss, CrossEntropySurvLoss


# ---------------  CLASSIFICATION LOOP  ------------------------
def train(datasets, args):
    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    
    loss_fn = nn.CrossEntropyLoss()
    
    print('Done!')
    
    print('\nInit Model...', end=' ')
    
    if args.model_type == 'abmil':
        from models.ABMIL import DAttention
        model = DAttention(args.in_dim, args.n_classes, dropout=args.drop_out, act='relu')
    elif args.model_type == 'clam_sb':
        from models.CLAM import CLAM_SB
        model = CLAM_SB(n_classes=args.n_classes, embed_dim=args.in_dim, dropout=args.drop_out, subtyping=True)
    elif args.model_type == 'trans_mil':
        from models.TransMIL import TransMIL
        model = TransMIL(args.in_dim, args.n_classes, dropout=args.drop_out, act='relu')
    elif args.model_type == 's4model':
        from models.S4MIL import S4Model
        model = S4Model(in_dim = args.in_dim, 
                        n_classes = args.n_classes, 
                        act = 'gelu', 
                        dropout = args.drop_out,
                        d_state = args.mambamil_state_dim,
                        d_model = args.mambamil_dim,
                        survival=args.survival)
    elif args.model_type == 'dsmil':
        from models.DSMIL import MILNet
        model = MILNet(in_size=args.in_dim, num_class=args.n_classes, dropout=args.drop_out)
    elif args.model_type == 'mamba_mil':
        from models.MambaMIL import MambaMIL
        model = MambaMIL(in_dim = args.in_dim, n_classes=args.n_classes, dropout=args.drop_out, act='gelu', layer = args.mambamil_layer, rate = args.mambamil_rate, type = args.mambamil_type)
    elif args.model_type == '2DMambaMIL':
        model = MambaMIL_2D(args=args)
    else:
        raise NotImplementedError(f'{args.model_type} is not implemented ...')
    
    model = model.to(device)
    optimizer = get_optim(model, args)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=0)

    train_loader = get_split_loader(train_split)
    val_loader = get_split_loader(val_split)
    test_loader = get_split_loader(test_split)
    print('Done!')

    early_stopping = None

    best_auc = 0

    result = {
        'best_model_acc': [0,0,0],
        'best_model_f1': [0,0,0],
        'best_model_auc': [0,0,0]
    }

    os.makedirs(f'{args.results_dir}/{args.task}_{args.exp_code}',exist_ok=True)

    for epoch in range(args.max_epochs):
        train_loop(epoch, model, train_loader, optimizer, scheduler, loss_fn, args)
        
        acc, f1, auc = validate(epoch, model, val_loader, loss_fn, args)

        t_accuracy, t_f1, t_auc = summary(model, epoch, test_loader, args)

        torch.save(model.state_dict(), os.path.join(args.results_dir, "checkpoint_{}.pt".format(epoch)))
        
        if auc >= best_auc:   # update based on val AUC
            best_auc = auc
            result['best_model_auc'] = [t_accuracy, t_f1, t_auc]
            torch.save(model, f'{args.results_dir}/{args.task}_{args.exp_code}/{args.model_type}_best_ckpt.pt')
        
        wandb.log({"AUC_acc": result['best_model_auc'][0], 
                "AUC_f1": result['best_model_auc'][1], 
                "AUC_auc": result['best_model_auc'][2]})

# TRAIN LOOP
def train_loop(epoch, model, loader, optimizer, scheduler, loss_fn = None, args=None):   
    device=torch.device(f'cuda:{args.device}')
    model.train()
    train_loss = 0.
    train_error = 0.

    ground_truth_list = []
    prediction_list = []
    prob_list = []

    print('\n')

    progress = tqdm(total=len(loader), desc=f"Train - Epoch {epoch+1}")

    for _, batch_data in enumerate(loader):
        data, coords, label = batch_data
        data = data.to(device).squeeze(0)
        coords = coords.to(device).squeeze(0)
        label = label.to(device)

        if args.model_type == '2DMambaMIL':
            logits, Y_prob, Y_hat, _, _ = model(data, coords)
        else:
            logits, Y_prob, Y_hat, _, _ = model(data)
        
        ground_truth_list.append(label.item())
        prediction_list.append(Y_hat.item())
        prob_list.append(Y_prob.tolist())

        loss = loss_fn(logits, label)
        loss_value = loss.item()
        train_loss += loss_value
        error = calculate_error(Y_hat, label)
        train_error += error
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        progress.update()
    scheduler.step()
    
    progress.close()
    
    prob_list_new = []
    for prob in prob_list:
        prob_list_new.append(prob[0])

    ground_truth_arr = []
    for label in ground_truth_list:
        binary_label = [0] * args.n_classes
        binary_label[label] = 1
        ground_truth_arr.append(binary_label)

    accuracy, f1, auc = calculate_metrics(prediction_list, prob_list_new, ground_truth_list, ground_truth_arr)

    train_loss /= len(loader)
    train_error /= len(loader)

    wandb.log({"train_acc": accuracy,
                "train_f1": f1, 
                "train_auc": auc,
                "train_loss": train_loss,
                "train_error": train_error}, step=epoch)
    print(f'train_acc: {accuracy}, train_f1: {f1}, train_auc: {auc}')

# VAL LOOP 
def validate(epoch, model, loader, loss_fn = None, args=None):
    device=torch.device(f'cuda:{args.device}')
    model.eval()
    val_loss = 0.
    val_error = 0.
    
    ground_truth_list = []
    prediction_list = []
    prob_list = []

    with torch.no_grad():
        progress = tqdm(total=len(loader), desc=f"Valid - Epoch {epoch+1}")
        for _, batch_data in enumerate(loader):
            data, coords, label = batch_data
            data = data.to(device).squeeze(0)
            coords = coords.to(device).squeeze(0)
            label = label.to(device)
            if args.model_type == '2DMambaMIL':
                logits, Y_prob, Y_hat, _, _ = model(data, coords)
            else:
                logits, Y_prob, Y_hat, _, _ = model(data)
        
            
            loss = loss_fn(logits, label)

            ground_truth_list.append(label.item())
            prediction_list.append(Y_hat.item())
            prob_list.append(Y_prob.tolist())
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            progress.update()
    
    progress.close()

    ground_truth_arr = []
    for label in ground_truth_list:
        binary_label = [0] * args.n_classes
        binary_label[label] = 1
        ground_truth_arr.append(binary_label)

    prob_list_new = []
    for prob in prob_list:
        prob_list_new.append(prob[0])

    accuracy, f1, auc = calculate_metrics(prediction_list, prob_list_new, ground_truth_list, ground_truth_arr)

    val_error /= len(loader)
    val_loss /= len(loader)

    wandb.log({"val_acc": accuracy, 
                   "val_f1": f1, 
                   "val_auc": auc,
                   "val_error": val_error,
                   "val_loss": val_loss}, step=epoch)
    print(f'epoch: {epoch + 1}, val_acc: {accuracy}, val_f1: {f1}, val_auc: {auc}')

    return accuracy, f1, auc

# TEST LOOP 
def summary(model, epoch, loader, args):
    device=torch.device(f'cuda:{args.device}')
    model.eval()
    
    ground_truth_list = []
    prediction_list = []
    prob_list = []

    with torch.no_grad():
        progress = tqdm(total=len(loader), desc=f"Test")
        for _, batch_data in enumerate(loader):
            data, coords, label = batch_data
            data = data.to(device).squeeze(0)
            coords = coords.to(device).squeeze(0)

            label = label.to(device)
            if args.model_type == '2DMambaMIL':
                logits, Y_prob, Y_hat, _, _ = model(data, coords)
            else:
                logits, Y_prob, Y_hat, _, _ = model(data)
        
    
            ground_truth_list.append(label.item())
            prediction_list.append(Y_hat.item())
            prob_list.append(Y_prob.tolist())
            
            progress.update()
    
    progress.close()
    
    ground_truth_arr = []
    for label in ground_truth_list:
        binary_label = [0] * args.n_classes
        binary_label[label] = 1
        ground_truth_arr.append(binary_label)

    prob_list_new = []
    for prob in prob_list:
        prob_list_new.append(prob[0])

    accuracy, f1, auc = calculate_metrics(prediction_list, prob_list_new, ground_truth_list, ground_truth_arr)
    wandb.log({"test_acc": accuracy, 
                   "test_f1": f1, 
                   "test_auc": auc}, step=epoch)
    print(f'TEST: test_acc: {accuracy}, test_f1: {f1}, test_auc: {auc}')
    
    return accuracy, f1, auc


# ---------------  SURVIVAL ANALYSI LOOP  ------------------------
def train_survival(datasets, args):
    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    print('\nInit loss function...', end=' ')
    
    if args.bag_loss == 'ce_surv':
        loss_fn = CrossEntropySurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'nll_surv':  # default
        loss_fn = NLLSurvLoss(alpha=0)
    elif args.bag_loss == 'cox_surv':
        loss_fn = CoxSurvLoss()
    else:
        raise NotImplementedError

    reg_fn = None

    print('Done!')
    
    print('\nInit Model...', end=' ')
    args.fusion = None 

    if args.model_type == 'abmil':
        from models.ABMIL import DAttention
        model = DAttention(args.in_dim, args.n_classes, dropout=args.drop_out, act='relu')
    elif args.model_type == 'clam_sb':
        from models.CLAM import CLAM_SB
        model = CLAM_SB(n_classes=args.n_classes, embed_dim=args.in_dim, dropout=args.drop_out, subtyping=True)
    elif args.model_type == 'trans_mil':
        from models.TransMIL import TransMIL
        model = TransMIL(args.in_dim, args.n_classes, dropout=args.drop_out, act='relu')
    elif args.model_type == 's4model':
        from models.S4MIL import S4Model
        model = S4Model(in_dim = args.in_dim, 
                        n_classes = args.n_classes, 
                        act = 'gelu', 
                        dropout = args.drop_out,
                        d_state = args.mambamil_state_dim,
                        d_model = args.mambamil_dim,
                        survival=args.survival)
    elif args.model_type == 'dsmil':
        from models.DSMIL import MILNet
        model = MILNet(in_size=args.in_dim, num_class=args.n_classes, dropout=args.drop_out)
    elif args.model_type == 'mamba_mil':
        from models.MambaMIL import MambaMIL
        model = MambaMIL(in_dim = args.in_dim, n_classes=args.n_classes, dropout=args.drop_out, act='gelu', layer = args.mambamil_layer, rate = args.mambamil_rate, type = args.mambamil_type)
    elif args.model_type == '2DMambaMIL':
        model = MambaMIL_2D(args=args)
    else:
        raise NotImplementedError(f'{args.model_type} is not implemented ...')
    
    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device('cuda'))
    
    print('Done!')
  
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader_survival(train_split, training=True, testing = args.testing, 
        weighted = args.weighted_sample, mode=args.mode, batch_size=1)
    val_loader = get_split_loader_survival(val_split, testing = args.testing, mode=args.mode, batch_size=1)
    print('Done!')

    max_val_c_index = 0

    model = model.to(device)
    optimizer = get_optim(model, args)

    os.makedirs(f'{args.results_dir}/{args.task}_{args.exp_code}',exist_ok=True)

    for epoch in range(args.max_epochs):
        train_loop_survival(epoch, model, train_loader, optimizer, loss_fn=loss_fn, reg_fn=reg_fn, args=args)
        c_index = validate_survival(epoch, model, val_loader, loss_fn=loss_fn, args=args)
        if c_index > max_val_c_index:
            max_val_c_index = c_index
            torch.save(model, f'{args.results_dir}/{args.task}_{args.exp_code}/{args.model_type}_best_ckpt.pt')
    
    print(args.model_type, args.task, args.fold)
    print('max_val_c_index: {:.4f}'.format(max_val_c_index))

    wandb.log({"max_val_c_index": max_val_c_index})

    print('Done!')

# TRAIN LOOP
def train_loop_survival(epoch, model, loader, optimizer, loss_fn=None, reg_fn=None, lambda_reg=1e-4, gc=32, args=None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    train_loss_surv, train_loss = 0., 0.

    print('\n')
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, batch in enumerate(loader):
        data_WSI, coords, data_omic, label, event_time, c = batch
        data_WSI, data_omic = data_WSI.to(device, non_blocking = True), data_omic.to(device, non_blocking = True)
        label = label.to(device, non_blocking = True)
        coords = coords.to(device, non_blocking = True)
        c = c.to(device, non_blocking=True)
        
        if args.model_type == '2DMambaMIL':
            hazards, S, _, _, _ = model(data_WSI, coords)
        else:
            hazards, S, _, _, _ = model(data_WSI)
        loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time

        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg
        loss = loss / gc + loss_reg
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    train_loss_surv /= len(loader)
    train_loss /= len(loader)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss_surv, train_loss, c_index))

# VALID LOOP
def validate_survival(epoch, model, loader, loss_fn=None, reg_fn=None, lambda_reg=1e-4, args=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, (data_WSI, coords, data_omic, label, event_time, c) in enumerate(loader):
        data_WSI, data_omic = data_WSI.to(device), data_omic.to(device)
        coords = coords.to(device)
        label = label.to(device)
        c = c.to(device)

        with torch.no_grad():
            if args.model_type == '2DMambaMIL':
                hazards, S, _, _, _ = model(data_WSI, coords)
            else:
                hazards, S, _, _, _ = model(data_WSI)
        loss = loss_fn(hazards=hazards, S=S, Y=label, c=c, alpha=0)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = -torch.sum(S, dim=1).cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.cpu().numpy()
        all_event_times[batch_idx] = event_time

        val_loss_surv += loss_value
        val_loss += loss_value + loss_reg

    val_loss_surv /= len(loader)
    val_loss /= len(loader)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    print('Epoch: {}, val_loss_surv: {:.4f}, val_loss: {:.4f}, val_c_index: {:.4f}'.format(epoch, val_loss_surv, val_loss, c_index))

    return c_index