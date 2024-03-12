import argparse
import os
import wandb
import sys
from timeit import default_timer as timer

import numpy as np
import pandas as pd

### Internal Imports
from datasets.dataset_survival import Generic_MIL_Survival_Dataset
from utilities.core_utils import train
from utilities.utils import get_custom_exp_code
from utilities.file_utils import save_pkl

### PyTorch Imports
import torch


def main(args):
	#### Create Results Directory
	if not os.path.isdir(args.results_dir):
		os.mkdir(args.results_dir)

	if args.k_start == -1:
		start = 0
	else:
		start = args.k_start
	if args.k_end == -1:
		end = args.k
	else:
		end = args.k_end

	latest_val_acc = []
	latest_val_auroc = []
	folds = np.arange(start, end) #[0, 1, 2, 3, 4]

	### Start 5-Fold CV Evaluation. 
	for i in folds:
		start = timer()
		seed_torch(args.seed)
		#./results/5foldcv/param_code/exp_code_s{seed}/split_latest_val_i_results.pkl
		results_pkl_path = os.path.join(args.results_dir, 'split_latest_val_{}_results.pkl'.format(i)) #0, ..., 5
		if os.path.isfile(results_pkl_path):
			print("Skipping Split %d" % i)
			continue

		### Gets the Train + Val Dataset Loader.
		train_dataset, val_dataset = dataset.return_splits(from_id=False, 
				csv_path='{}/splits_{}.csv'.format(args.split_dir, i)) #train_split, val_split
		
		print('training: {}, validation: {}'.format(len(train_dataset), len(val_dataset))) #training: 308, validation: 77
		datasets = (train_dataset, val_dataset)
		
		### Specify the input dimension size if using genomic features.
		if 'omic' in args.mode:
			args.omic_input_dim = train_dataset.genomic_features.shape[1]
			print("Genomic Dimension", args.omic_input_dim)
		elif 'coattn' in args.mode:
			args.omic_sizes = train_dataset.omic_sizes #array with the length of each category, ex [94, 334, 521, 468, 1496, 479]
			print('Genomic Dimensions', args.omic_sizes) #[82, 323, 510, 431, 1461, 444]
		else:
			args.omic_input_dim = 0

		### Run Train-Val on Survival Task.
		if args.task_type == 'survival':
			val_latest, acc, auroc = train(datasets, i, args)
			latest_val_acc.append(acc)
			latest_val_auroc.append(auroc)

		### Write Results for Each Split to PKL
		save_pkl(results_pkl_path, val_latest)
		end = timer()
		print('Fold %d Time: %f seconds' % (i, end - start))

	### Finish 5-Fold CV Evaluation.
	if args.task_type == 'survival':
		results_latest_df = pd.DataFrame({'folds': folds, 'val_acc': latest_val_acc, 'val_auroc': latest_val_auroc})

	results_latest_df.to_csv(os.path.join(args.results_dir, 'summary_latest.csv'))

### Training settings
parser = argparse.ArgumentParser(description='Configurations for Survival Analysis on TCGA Data.')
parser.add_argument('--data_root_dir',   type=str, default='/work/ai4bio2023/Data/WSI_Data/Features', help='Data directory to WSI features (extracted via CLAM')
parser.add_argument('--seed', 			 type=int, default=1, help='Random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', 			     type=int, default=5, help='Number of folds (default: 5)')
parser.add_argument('--k_start',		 type=int, default=-1, help='Start fold (Default: -1, last fold)')
parser.add_argument('--k_end',			 type=int, default=-1, help='End fold (Default: -1, first fold)')
parser.add_argument('--results_dir',     type=str, default='./results', help='Results directory (Default: ./results)')
parser.add_argument('--which_splits',    type=str, default='5foldcv', help='Which splits folder to use in ./splits/ (Default: ./splits/5foldcv')
parser.add_argument('--split_dir',       type=str, default='tcga_ovary', help='Which cancer type within ./splits/<which_splits> to use for training')
parser.add_argument('--overwrite',     	 action='store_true', default=False, help='Whether or not to overwrite experiments (if already ran)')
parser.add_argument('--testing',         type=bool, default=False)
parser.add_argument('--num-classes',     type=int, default=2)
parser.add_argument('--thr',             type=int, default=42)

### Model Parameters.
parser.add_argument('--model_type',      type=str, choices=['snn', 'mcat', 'mcat_vit', 'mcat_hvit'], default='mcat_vit', help='Type of model (Default: mcat)')
parser.add_argument('--mode',            type=str, choices=['omic', 'coattn'], default='coattn', help='Specifies which modalities to use / collate function in dataloader.')
parser.add_argument('--fusion',          type=str, choices=['None', 'concat', 'bilinear'], default='None', help='Type of fusion. (Default: concat).')
parser.add_argument('--apply_sig',		 action='store_true', default=True, help='Use genomic features as signature embeddings.')
parser.add_argument('--drop_out',        action='store_true', default=True, help='Enable dropout (p=0.25)')
parser.add_argument('--model_size_wsi',  type=str, default='small', help='Network size of AMIL model')
parser.add_argument('--model_size_omic', type=str, default='small', help='Network size of SNN model')
parser.add_argument('--met',             type=bool, default=True, help='Use met data')
parser.add_argument('--visual_dropout',  type=float, default=0, help='visual dropout to mitigate WSI noise')

### Optimizer Parameters + Survival Loss Function
parser.add_argument('--opt',             type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--batch_size',      type=int, default=1, help='Batch Size (Default: 1, due to varying bag sizes)')
parser.add_argument('--gc',              type=int, default=32, help='Gradient Accumulation Step.')
parser.add_argument('--max_epochs',      type=int, default=20, help='Maximum number of epochs to train (default: 20)')
parser.add_argument('--lr',				 type=float, default=2e-4, help='Learning rate (default: 0.0001)')
parser.add_argument('--bag_loss',        type=str, default='ce', help='slide-level classification loss function (default: ce)')
parser.add_argument('--reg', 			 type=float, default=1e-5, help='L2-regularization weight decay (default: 1e-5)')
parser.add_argument('--reg_type',        type=str, choices=['None', 'omic'], default='None', help='Which network submodules to apply L1-Regularization (default: None)')
parser.add_argument('--lambda_reg',      type=float, default=1e-4, help='L1-Regularization Strength (Default 1e-4)')
parser.add_argument('--weighted_sample', action='store_true', default=True, help='Enable weighted sampling')
parser.add_argument('--early_stopping',  action='store_true', default=True, help='Enable early stopping')

args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = get_custom_exp_code(args)
args.task = '_'.join(args.split_dir.split('_')[:2]) + '_survival' # tcga_ovary_survival
print("Experiment Name:", args.exp_code)

### Sets Seed for reproducible experiments.
def seed_torch(seed=7):
	import random
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if device.type == 'cuda':
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 1024
settings = {'num_splits': args.k, 
			'k_start': args.k_start,
			'k_end': args.k_end,
			'task': args.task,
			'max_epochs': args.max_epochs, 
			'results_dir': args.results_dir, 
			'lr': args.lr,
			'experiment': args.exp_code,
			'reg': args.reg,
			'bag_loss': args.bag_loss,
			'seed': args.seed,
			'model_type': args.model_type,
			'model_size_wsi': args.model_size_wsi,
			'model_size_omic': args.model_size_omic,
			"use_drop_out": args.drop_out,
			'weighted_sample': args.weighted_sample,
			'gc': args.gc,
			'opt': args.opt,
			'visual_dropout': args.visual_dropout,
			'thr': args.thr,	
	}
print('\nLoad Dataset')

if 'survival' in args.task:
	study = '_'.join(args.task.split('_')[:2])  #tcga_ovary
	dataset = Generic_MIL_Survival_Dataset(csv_path = '%s/%s_all_clean_%s.csv' % (args.dataset_path, study, args.met),
										num_classes=args.num_classes,
										mode = args.mode,
										apply_sig = args.apply_sig,
										data_dir= args.data_root_dir, #path for WSI features 
										shuffle = False, 
										seed = args.seed, 
										print_info = True,
										patient_strat= False,
										label_col = 'label',
										met=args.met,
										thr=args.thr,
										)
else:
	raise NotImplementedError

if isinstance(dataset, Generic_MIL_Survival_Dataset):
	args.task_type = 'survival'
else:
	raise NotImplementedError

### Creates results_dir Directory.
if not os.path.isdir(args.results_dir):
	os.mkdir(args.results_dir)

### Appends to the results_dir path: 1) which splits were used for training (e.g. - 5foldcv), and then 2) the parameter code and 3) experiment code
args.results_dir = os.path.join(args.results_dir, args.which_splits, args.param_code, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
	os.makedirs(args.results_dir)

if ('summary_latest.csv' in os.listdir(args.results_dir)) and (not args.overwrite):
	print("Exp Code <%s> already exists! Exiting script." % args.exp_code)
	sys.exit()

### Sets the absolute path of split_dir: ./splits/5foldcv/tcga_ovary
args.split_dir = os.path.join('./splits', args.which_splits, args.split_dir) 
print("split_dir", args.split_dir)
assert os.path.isdir(args.split_dir)
settings.update({'split_dir': args.split_dir})

#Create the file ./results/5foldcv/param_code/exp_code_s{seed}/experiment_{exp_code}.txt and write settings
with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f: 
	print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
	print("{}:  {}".format(key, val))        

if __name__ == "__main__":
	start = timer()
	wandb.init(
        project="MCAT",
        name='Survival_Prediction',
        config={
        "which_splits": args.which_splits,
		"k": args.k,
        "learning_rate": args.lr,
        "model_type": args.model_type,
        "epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "fusion": args.fusion,
        "mode": args.mode,
        "apply_sig":args.apply_sig,
		"model_size_wsi":args.model_size_wsi,
		"model_size_omic": args.model_size_omic,
		"bag_loss": args.bag_loss,
		"met": args.met,
		"gc": args.gc,
		"early_stopping": args.early_stopping,
		"visual_dropout": args.visual_dropout,
		"thr": args.thr,
        }
    )

	results = main(args)
	end = timer()
	wandb.finish()
	print("finished!")
	print("end script")
	print('Script Time: %f seconds' % (end - start))
