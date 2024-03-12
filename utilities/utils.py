from __future__ import print_function, division
import torch
import numpy as np
import numpy as np
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler
import torch.optim as optim

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def collate_MIL_survival(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    omic = torch.cat([item[1] for item in batch], dim = 0).type(torch.FloatTensor)
    label = torch.LongTensor([item[2] for item in batch])
    return [img, omic, label]

def collate_MIL_survival_sig(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    omic1 = torch.cat([item[1] for item in batch], dim = 0).type(torch.FloatTensor)
    omic2 = torch.cat([item[2] for item in batch], dim = 0).type(torch.FloatTensor)
    omic3 = torch.cat([item[3] for item in batch], dim = 0).type(torch.FloatTensor)
    omic4 = torch.cat([item[4] for item in batch], dim = 0).type(torch.FloatTensor)
    omic5 = torch.cat([item[5] for item in batch], dim = 0).type(torch.FloatTensor)
    omic6 = torch.cat([item[6] for item in batch], dim = 0).type(torch.FloatTensor)

    label = torch.LongTensor([item[7] for item in batch])
    return [img, omic1, omic2, omic3, omic4, omic5, omic6, label]


def get_split_loader(split_dataset, training = False, testing = False, weighted = False, mode='coattn', batch_size=1):
    """
        return either the validation loader or training loader 
    """
    if mode == 'coattn':
        collate = collate_MIL_survival_sig
    else:
        collate = collate_MIL_survival

    kwargs = {'num_workers': 4} if device.type == "cuda" else {}
    if not testing:
        if training:
            if weighted:
                weights = make_weights_for_balanced_classes_split(split_dataset)
                loader = DataLoader(split_dataset, batch_size=batch_size, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate, **kwargs)    
            else:
                loader = DataLoader(split_dataset, batch_size=batch_size, sampler = RandomSampler(split_dataset), collate_fn = collate, **kwargs)
        else:
            loader = DataLoader(split_dataset, batch_size=batch_size, sampler = SequentialSampler(split_dataset), collate_fn = collate, **kwargs)
    
    else:
        # Generate 10% of the indices from the length of split_dataset 
        ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace = False)
        loader = DataLoader(split_dataset, batch_size=1, sampler = SubsetSequentialSampler(ids), collate_fn = collate, **kwargs )

    return loader

def get_optim(model, args):
    if args.opt == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    else:
        raise NotImplementedError
    return optimizer

def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)
    
    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n
    
    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)

#samples: Total number of samples.
#cls_ids: A list containing indices of samples for each class.
def generate_split(cls_ids, val_num, slide_data, samples, filename, n_splits = 5, seed = 7):

    indices = np.arange(samples).astype(int)
    np.random.seed(seed)

    for i in range(n_splits): #5
        all_val_ids = []
        sampled_train_ids = []

        #validation indeces
        for c in range(len(val_num)): #2
            """
            For each split, it samples validation and test indices for each class, ensuring that the sampled indices are not already in the validation or test sets.
            """
            #cls_ids[c]: array containing the indices of samples belonging to class c.
            possible_indices = np.intersect1d(cls_ids[c], indices) #all indices of this class
            remaining_ids = possible_indices

            if val_num[c] > 0:
                val_ids = np.random.choice(possible_indices, val_num[c], replace = False) # validation ids
                remaining_ids = np.setdiff1d(possible_indices, val_ids) #indices of this class left after validation
                all_val_ids.append(sorted(val_ids))
                sampled_train_ids.append(sorted(remaining_ids))
     
                
        sampled_train_ids = [elemento for sublist in sampled_train_ids for elemento in sublist] #308
        all_val_ids = [elemento for sublist in all_val_ids for elemento in sublist] #77
        slideids_train = slide_data.iloc[sampled_train_ids]['case_id'].reset_index() #case:id
        slideids_val = slide_data.iloc[all_val_ids]['case_id'].reset_index()
        slideids_val, slideids_train = slideids_val.align(slideids_train, fill_value=None)

        df = pd.DataFrame({'train': slideids_train['case_id'], 'val': slideids_val['case_id']})
        fname = f"{filename}_{i}.csv"
        df.to_csv(fname, index=True)   

def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))                                           
    weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]  #peso = n sample/n sample della classe = [1.7906976744186047, 2.264705882352941]                                                                                               
    weight = [0] * int(N)                                           
    for idx in range(len(dataset)):   
        y = int(dataset.getlabel(idx))                        
        weight[idx] = weight_per_class[y]  #per ogni sample gli assegna il suo peso (stabilito dalla sua classe)                                

    return torch.DoubleTensor(weight)


def l1_reg_all(model, reg_type=None):
    l1_reg = None

    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)
    return l1_reg

def get_custom_exp_code(args):
    exp_code = '_'.join(args.split_dir.split('_')[:2]) #tcga_ovary
    dataset_path = 'dataset_csv'
    param_code = ''

    ### Model Type
    if args.model_type == 'snn':
        param_code += 'SNN'
    elif args.model_type == 'mcat':
        param_code += 'MCAT'
    elif args.model_type == 'mcat_vit':
        param_code += 'MCAT_ViT'
    elif args.model_type == 'mcat_hvit':
        param_code += 'MCAT_HViT'
    else:
      raise NotImplementedError
    
    if args.met:
        param_code+='_met'

    ### Loss Function
    param_code += '_%s' % args.bag_loss

    ### Learning Rate
    if args.lr != 2e-4:
      param_code += '_lr%s' % format(args.lr, '.0e')

    ### L1-Regularization
    if args.reg_type != 'None':
      param_code += '_reg%s' % format(args.lambda_reg, '.0e')

    param_code += '_%s' % args.which_splits.split("_")[0]

    ### Batch Size
    if args.batch_size != 1:
      param_code += '_b%s' % str(args.batch_size)

    ### Gradient Accumulation
    if args.gc != 1:
        param_code += '_gc%s' % str(args.gc) 

    if args.visual_dropout:
        param_code += '_vd%s' % str(args.visual_dropout) 
        

    ### Fusion Operation
    if args.fusion != "None":
      param_code += '_' + args.fusion

    args.exp_code = exp_code + "_" + param_code
    args.param_code = param_code
    args.dataset_path = dataset_path

    return args