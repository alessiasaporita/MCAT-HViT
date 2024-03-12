
import pandas as pd
import numpy as np
from utilities.utils import generate_split

"""
Function to generate 5 splits with:
80% training, 20% validation samples for each class: dead=113/28 alive=94/23, without met -->258
80% samples for training and 20% samples for validation ---> 207 train, 51 val
"""

def cls_ids_prep(slide_data, num_classes):
    slide_cls_ids = [[] for i in range(num_classes)] #[[], []]
    for i in range(num_classes):
        slide_cls_ids[i] = np.where(slide_data['label'] == i)[0]
    return slide_cls_ids 


if __name__ == "__main__":
    slide_data = pd.read_csv('/work/ai4bio2023/Data/tcga_ovary_all_clean_False.csv') #csv path
    num_classes = 2
    slide_data = slide_data[(slide_data['vital_status'] == 0) | ((slide_data['vital_status'] == 1) & (slide_data['survival_months'] > 42))]
    slide_data.reset_index(drop=True, inplace=True)
    for i in slide_data.index:
        if ((slide_data['vital_status'][i] == 0 and slide_data['survival_months'][i] > 42) or (slide_data['vital_status'][i] == 1)):
            slide_data.at[i, 'label'] = 1  # alive
        else:
            slide_data.at[i, 'label'] = 0  # dead
    new_cols = list(slide_data.columns[-1:]) + list(slide_data.columns[:-2]) #change order of the columns 
    slide_data = slide_data[new_cols]
    slide_cls_ids = cls_ids_prep(slide_data, num_classes)
    val_num = (28, 23) #2 classes, Dead = 0, O Alive = 1 ---> computed manually
    filename = '/work/ai4bio2023/MCAT-HViT/splits/5foldcv/tcga_ovary/splits' #folder filename
    generate_split(slide_cls_ids, val_num, slide_data, len(slide_data), filename)
   


