import numpy as np
import torch
from utilities.utils import *
import wandb
from utilities.metrics import Accuracy, AUROC
import torch.nn.functional as F


def train_loop_survival_coattn(epoch, model, loader, optimizer, n_classes, loss_fn=None, reg_fn=None, lambda_reg=0., gc=16):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    accuracy = Accuracy()
    auroc=AUROC()
    train_loss_surv, train_loss = 0., 0.

    print('\n')
    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label) in enumerate(loader):

        data_WSI = data_WSI.to(device) #(n_patches, 1024)
        data_omic1 = data_omic1.type(torch.FloatTensor).to(device) #82, 146
        data_omic2 = data_omic2.type(torch.FloatTensor).to(device) #323, 555
        data_omic3 = data_omic3.type(torch.FloatTensor).to(device) #510, 865
        data_omic4 = data_omic4.type(torch.FloatTensor).to(device) #431, 736
        data_omic5 = data_omic5.type(torch.FloatTensor).to(device) #1461, 2424
        data_omic6 = data_omic6.type(torch.FloatTensor).to(device) #444, 733
        label = label.type(torch.LongTensor).to(device) #1

        logits, A  = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)
        loss = F.cross_entropy(logits, label)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        accuracy.update(logits, label)
        auroc.update(logits, label)

        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg

        if (batch_idx + 1) % 100 == 0:
            print('batch {}, loss: {:.4f}, label: {}'.format(batch_idx, loss_value + loss_reg, label.item()))
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
    print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}'.format(epoch, train_loss_surv, train_loss))
    wandb.log({'train/loss_surv': train_loss_surv, 'train/loss': train_loss, 'train/acc': acc, 'train/aur': aur})


def validate_survival_coattn(cur, epoch, model, loader, n_classes, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    accuracy = Accuracy()
    auroc=AUROC()

    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label) in enumerate(loader):

        data_WSI = data_WSI.to(device)
        data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
        data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
        data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
        data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
        data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
        data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
        label = label.type(torch.LongTensor).to(device)

        with torch.no_grad():
            logits, A = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6) # return hazards, S, Y_hat, A_raw, results_dict

        loss = F.cross_entropy(logits, label)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg
  
        val_loss_surv += loss_value
        val_loss += loss_value + loss_reg

        accuracy.update(logits, label)
        auroc.update(logits, label)


    val_loss_surv /= len(loader)
    val_loss /= len(loader)
    acc = accuracy.compute()
    aur = auroc.compute()
    wandb.log({'val/loss_surv': val_loss_surv, 'val/loss': val_loss, 'val/acc': acc, 'val/aur': aur}) 
    return aur.item()


def summary_survival_coattn(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    accuracy = Accuracy()
    auroc=AUROC()
    case_ids = loader.dataset.slide_data['case_id'] 
    patient_results = {}

    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label) in enumerate(loader):
        
        data_WSI = data_WSI.to(device)
        data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
        data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
        data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
        data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
        data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
        data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
        label = label.type(torch.LongTensor).to(device)
        
        case_id = case_ids.iloc[batch_idx]

        with torch.no_grad():
            logits, A  = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6) # return hazards, S, Y_hat, A_raw, results_dict
        
        accuracy.update(logits, label)
        auroc.update(logits, label)
        patient_results.update({case_id: {'case_id': np.array(case_id), 'logits': logits, 'label': label}})

    acc = accuracy.compute()
    aur = auroc.compute()
    return patient_results, acc.item(), aur.item()