from argparse import ArgumentParser
import torch
import os
import pandas as pd
import numpy as np

"""
function to check the average number of patches of WSIs
"""
def check_WSI(args):
    slide_data = pd.read_csv(args.csv_path, low_memory=False)
    wsi_dim=[]
    slide_ids = slide_data['slide_id']
    for slide_id in slide_ids:
        wsi_path = os.path.join(args.data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
        wsi_bag = torch.load(wsi_path)
        wsi_dim.append(wsi_bag.shape[0])
    wsi_dim = np.array(wsi_dim)
    mean = wsi_dim.mean()
    print("Average patches WSI:{}".format(mean))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_dir", default='/work/ai4bio2023/Data/WSI_Data/Features', type=str) 
    parser.add_argument("--csv_path", default='/work/ai4bio2023/MCAT/dataset_csv/tcga_ovary_all_clean_False.csv', type=str) 
    args = parser.parse_args()
    check_WSI(args)



