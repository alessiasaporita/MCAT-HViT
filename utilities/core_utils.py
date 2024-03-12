from argparse import Namespace
import os
import wandb
import numpy as np
from utilities.metrics import AUROC, Accuracy
import torch
from models.model_genomic import SNN
from models.model_coattn import MCAT_Surv
from models.model_mcat_hvit import MCAT_HViT_Surv
from models.model_mcat_vit import MCAT_ViT_Surv
from utilities.utils import l1_reg_all
from utilities.coattn_train_utils import *


def train(datasets: tuple, cur: int, args: Namespace):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur)) #cur = i, where i = 0, ..., 4
    train_split, val_split = datasets
 
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    print('\nInit loss function...', end=' ')
    if args.task_type == 'survival':
        if args.bag_loss == 'ce':
            loss_fn=None
            print("Using CrossEntropy\n")
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    
    #regularization
    if args.reg_type == 'omic': #default None
        reg_fn = l1_reg_all
    else:
        reg_fn = None

    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.num_classes} #True, 
    args.fusion = None if args.fusion == 'None' else args.fusion #concat

    #only genomic features
    if args.model_type =='snn':
        model_dict = {'input_dim': args.omic_input_dim, 'model_size_omic': args.model_size_omic, 'n_classes': args.num_classes}
        model = SNN(**model_dict) 
    #MCAT 
    elif args.model_type == 'mcat':
        model_dict = {'fusion': args.fusion, 'omic_sizes': args.omic_sizes, 'n_classes': args.num_classes}
        model = MCAT_Surv(**model_dict)
    elif args.model_type == 'mcat_hvit':
        model_dict = {'omic_sizes': args.omic_sizes, 'n_classes': args.num_classes, 'visual_dropout': args.visual_dropout}
        model = MCAT_HViT_Surv(**model_dict)
    elif args.model_type == 'mcat_vit':
        model_dict = {'omic_sizes': args.omic_sizes, 'n_classes': args.num_classes, 'visual_dropout': args.visual_dropout}
        model = MCAT_ViT_Surv(**model_dict)
    else:
        raise NotImplementedError
    
    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device('cuda'))
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, 
        weighted = args.weighted_sample, mode=args.mode, batch_size=args.batch_size) #dataloader
    val_loader = get_split_loader(val_split,  testing = args.testing, mode=args.mode, batch_size=args.batch_size)
    print('Done!')

    best_metric = float("-inf")
    for epoch in range(args.max_epochs):
        if args.task_type == 'survival':
            if args.mode == 'coattn':
                train_loop_survival_coattn(epoch, model, train_loader, optimizer, args.num_classes, loss_fn, reg_fn, args.lambda_reg, args.gc)
                metric = validate_survival_coattn(cur, epoch, model, val_loader, args.num_classes, loss_fn, reg_fn, args.lambda_reg, args.results_dir)
            else:
                train_loop_survival(epoch, model, train_loader, optimizer, args.num_classes, loss_fn, reg_fn, args.lambda_reg, args.gc)
                metric = validate_survival(cur, epoch, model, val_loader, args.num_classes, loss_fn, reg_fn, args.lambda_reg, args.results_dir)
        if args.early_stopping:
            if metric > best_metric:
                best_metric = metric
                torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    
    if not args.early_stopping:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    if args.mode == 'coattn':
        results_val_dict, acc, auroc = summary_survival_coattn(model, val_loader, args.num_classes)
    else:
        results_val_dict, acc, auroc = summary_survival(model, val_loader, args.num_classes)
    print('Validation acc: {:.4f}, Validation aur: {:.4f}'.format(acc, auroc))
    return results_val_dict, acc, auroc


def train_loop_survival(epoch, model, loader, optimizer, n_classes, loss_fn=None, reg_fn=None, lambda_reg=0., gc=16):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    accuracy = Accuracy()
    auroc=AUROC()
    train_loss_surv, train_loss = 0., 0.

    print('\n')
    for batch_idx, (data_WSI, data_omic, label) in enumerate(loader):
        data_WSI, data_omic = data_WSI.to(device), data_omic.to(device)
        label = label.to(device)

        logits = model(x_path=data_WSI, x_omic=data_omic) 
        loss = F.cross_entropy(logits, label)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg #l1_reg_all(model) * 0.0001

        accuracy.update(logits, label)
        auroc.update(logits, label)
        
        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg

        if (batch_idx + 1) % 100 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value + loss_reg, label.item(), data_WSI.size(0)))
        # backward pass
        loss = loss / gc + loss_reg
        loss.backward()

        if (batch_idx + 1) % gc == 0: 
            optimizer.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss_surv /= len(loader)
    train_loss /= len(loader)
    acc = accuracy.compute()
    aur = auroc.compute()
    print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_acc: {:.4f}, train_auroc: {:.4f}'.format(epoch, train_loss_surv, train_loss, acc.item(), aur.item()))
    wandb.log({'train/loss_surv': train_loss_surv, 'train/loss': train_loss, 'train/acc': acc.item(), 'train/aur': aur.item()})


def validate_survival(cur, epoch, model, loader, n_classes, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    accuracy = Accuracy()
    auroc=AUROC()

    for batch_idx, (data_WSI, data_omic, label) in enumerate(loader):
        data_WSI, data_omic = data_WSI.to(device), data_omic.to(device)
        label = label.to(device)

        with torch.no_grad():
            logits = model(x_path=data_WSI, x_omic=data_omic)
            loss = F.cross_entropy(logits, label)
            loss_value = loss.item()

            if reg_fn is None:
                loss_reg = 0
            else:
                loss_reg = reg_fn(model) * lambda_reg
            
        #loss_value = loss_value / gc
        accuracy.update(logits, label)
        auroc.update(logits, label)
        val_loss_surv += loss_value
        val_loss += loss_value + loss_reg

    val_loss_surv /= len(loader)
    val_loss /= len(loader)
    acc = accuracy.compute()
    aur = auroc.compute()

    print('Epoch: {}, val_loss_surv: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, val_auroc: {:.4f}'.format(epoch, val_loss_surv, val_loss, acc.item(), aur.item()))
    wandb.log({'val/loss_surv': val_loss_surv, 'val/loss': val_loss, 'val/acc': acc.item(), 'val/aur': aur.item()})
    return aur.item()


def summary_survival(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    accuracy = Accuracy()
    auroc=AUROC()

    case_ids = loader.dataset.slide_data['case_id']
    patient_results = {}

    for batch_idx, (data_WSI, data_omic, label) in enumerate(loader):
        data_WSI, data_omic = data_WSI.to(device), data_omic.to(device)
        label = label.to(device)
        case_id = case_ids.iloc[batch_idx]

        with torch.no_grad():
            logits = model(x_path=data_WSI, x_omic=data_omic)

        accuracy.update(logits, label)
        auroc.update(logits, label)
        patient_results.update({case_id: {'case_id': np.array(case_id), 'logits': logits, 'label': label}})

    acc = accuracy.compute()
    aur = auroc.compute()
    return patient_results, acc.item(), aur.item()