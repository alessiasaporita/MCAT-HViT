import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

class Generic_WSI_Survival_Dataset(Dataset):
    def __init__(self,
        csv_path = 'dataset_csv/ccrcc_clean.csv', num_classes = 2, mode = 'omic', apply_sig = False,
        shuffle = False, seed = 7, print_info = True, patient_strat=False, label_col = None, met=False, thr=42):
        """
        Generic_WSI_Survival_Dataset 

        Args:
            csv_file (string): Path to the csv file with annotations.
            shuffle (boolean): Whether to shuffle
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
            ignore (list): List containing class labels to ignore
        """
        self.custom_test_ids = None
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
        self.data_dir = None
        self.met = met

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)

        slide_data = pd.read_csv(csv_path, low_memory=False)
        if 'case_id' not in slide_data:
            NotImplementedError
        
        #patients alive for more than three years and a half or dead
        slide_data = slide_data[(slide_data['vital_status'] == 0) | ((slide_data['vital_status'] == 1) & (slide_data['survival_months'] > thr))]
        slide_data.reset_index(drop=True, inplace=True)
        for i in slide_data.index:
            if ((slide_data['vital_status'][i] == 0 and slide_data['survival_months'][i] > thr) or (slide_data['vital_status'][i] == 1)):
                slide_data.at[i, 'label'] = 1  # alive
            else:
                slide_data.at[i, 'label'] = 0  # dead

        new_cols = list(slide_data.columns[-1:]) + list(slide_data.columns[:-2]) #change order of the columns 
        slide_data = slide_data[new_cols]

        if not label_col:
            label_col = 'label' 
        else:
            assert label_col in slide_data.columns
        self.label_col = label_col

        patients_df = slide_data.drop_duplicates(['case_id']).copy()
        patient_dict = {} #dictionary with case_id: array with all slide_ids of the patient
        for patient in patients_df['case_id']:
            slide_ids = slide_data[slide_data['case_id']==patient]['slide_id'] 
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1) 
            else:
                slide_ids = slide_ids.values #all slide_ids of the patient
            patient_dict.update({patient:slide_ids})

        self.patient_dict = patient_dict 
        self.num_classes=num_classes 
        self.patient_data = {'case_id':patients_df['case_id'].values, 'label':patients_df['label'].values} #dict with 'case_id':array of all case_ids, 'label': array of all labels
        self.slide_data = slide_data
        self.metadata = slide_data.columns[:5] #all data that don't refer to genomic features 
        """
        ['label', 'case_id', 'slide_id', 'survival_months', 'vital_status']
        """
        self.mode = mode
        self.cls_ids_prep()

        if print_info:
            self.summarize()

        ### Signatures
        self.apply_sig = apply_sig
        if self.apply_sig:
            self.signatures = pd.read_csv('dataset_csv/signatures.csv')
        else:
            self.signatures = None


    def cls_ids_prep(self):
        self.patient_cls_ids = [[] for i in range(self.num_classes)] #[[], []]
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0] 

        self.slide_cls_ids = [[] for i in range(self.num_classes)] #[[], []]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0] 


    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])
        else:
            return len(self.slide_data)

    def summarize(self):
        print("label column: {}".format(self.label_col))
        print("number of classes: {}".format(self.num_classes))
        print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
        for i in range(self.num_classes):
            print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
            print('Slide-LVL; Number of samples registered in class  %d: %d' % (i, self.slide_cls_ids[i].shape[0]))


    def get_split_from_df(self, all_splits: dict, split_key: str='train', scaler=None):
        split = all_splits[split_key]  
        split = split.dropna().reset_index(drop=True) 

        if len(split) > 0:
            mask = self.slide_data['case_id'].isin(split.tolist()) 
            df_slice = self.slide_data[mask].reset_index(drop=True) 
            split = Generic_Split(df_slice, metadata=self.metadata, mode=self.mode, signatures=self.signatures, data_dir=self.data_dir, label_col=self.label_col, patient_dict=self.patient_dict, num_classes=self.num_classes, met=self.met)
        else:
            split = None
        
        return split


    def return_splits(self, from_id: bool=True, csv_path: str=None):
        if from_id:
            raise NotImplementedError
        else:
            assert csv_path 
            all_splits = pd.read_csv(csv_path)
            train_split = self.get_split_from_df(all_splits=all_splits, split_key='train') #Generic_Split
            val_split = self.get_split_from_df(all_splits=all_splits, split_key='val') #Generic_Split

            ### --> Normalizing Data
            print("****** Normalizing Data ******")
            train_scalers = train_split.get_scaler() #Standard Scaler
            train_split.apply_scaler(scalers=train_scalers)
            val_scalers = val_split.get_scaler() #Standard Scaler
            val_split.apply_scaler(scalers=val_scalers)
            ###
        return train_split, val_split

    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def __getitem__(self, idx):
        return None


class Generic_MIL_Survival_Dataset(Generic_WSI_Survival_Dataset):
    def __init__(self, data_dir, num_classes, mode: str='omic', **kwargs):
        super(Generic_MIL_Survival_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.mode = mode
        self.use_h5 = False
        self.num_classes = num_classes

    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def __getitem__(self, idx):
        case_id = self.slide_data['case_id'][idx]
        label = self.slide_data['label'][idx]
        slide_ids = self.patient_dict[case_id]
        data_dir = self.data_dir
        
        if not self.use_h5:
            if self.data_dir:
                if self.mode == 'omic':
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    return (torch.zeros((1,1)), genomic_features, label)

                elif self.mode == 'coattn':
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
                    omic1 = torch.tensor(self.genomic_features[self.omic_names[0]].iloc[idx])
                    omic2 = torch.tensor(self.genomic_features[self.omic_names[1]].iloc[idx])
                    omic3 = torch.tensor(self.genomic_features[self.omic_names[2]].iloc[idx])
                    omic4 = torch.tensor(self.genomic_features[self.omic_names[3]].iloc[idx])
                    omic5 = torch.tensor(self.genomic_features[self.omic_names[4]].iloc[idx])
                    omic6 = torch.tensor(self.genomic_features[self.omic_names[5]].iloc[idx])
                    return (path_features, omic1, omic2, omic3, omic4, omic5, omic6, label)

                else:
                    raise NotImplementedError('Mode [%s] not implemented.' % self.mode)
                ### <--
            else:
                return slide_ids, label


class Generic_Split(Generic_MIL_Survival_Dataset):
    #slide_data of the chosen split (0, .., 4) for training or validation
    def __init__(self, slide_data, metadata, mode, signatures=None, data_dir=None, label_col=None, patient_dict=None, num_classes=2, met=False, thr=42):
        self.use_h5 = False
        self.slide_data = slide_data
        self.metadata = metadata
        self.mode = mode
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.label_col = label_col
        self.patient_dict = patient_dict
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

        ### --> Initializing genomic features in Generic Split
        self.genomic_features = self.slide_data.drop(self.metadata, axis=1)
        self.signatures = signatures #file csv with the gene names belonging to the 6 categories 

        def series_intersection(s1, s2):
            return pd.Series(list(set(s1) & set(s2)))

        if self.signatures is not None:
            self.omic_names = []
            for col in self.signatures.columns:
                omic = self.signatures[col].dropna().unique()
                if met:
                    omic = np.concatenate([omic+mode for mode in ['_rnaseq', '_met']])
                else:
                    omic = np.concatenate([omic+mode for mode in ['_rnaseq']]) 
                omic = sorted(series_intersection(omic, self.genomic_features.columns)) 
                self.omic_names.append(omic) 
            self.omic_sizes = [len(omic) for omic in self.omic_names]
            #[82, 323, 510, 431, 1461, 444] without met
            #[146, 555, 865, 736, 2424, 733] with met

        print("Shape", self.genomic_features.shape)
        ### <--

    def __len__(self):
        return len(self.slide_data)

    ### --> Getting StandardScaler of self.genomic_features
    def get_scaler(self):
        scaler_omic = StandardScaler().fit(self.genomic_features)
        return (scaler_omic,)
    ### <--

    ### --> Applying StandardScaler to self.genomic_features
    def apply_scaler(self, scalers: tuple=None):
        transformed = pd.DataFrame(scalers[0].transform(self.genomic_features))
        transformed.columns = self.genomic_features.columns
        self.genomic_features = transformed
    ### <--