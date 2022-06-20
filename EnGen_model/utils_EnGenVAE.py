
import os
import sys
import time
import numpy as np
import torch
from datetime import datetime
from sklearn import preprocessing
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import pickle




class GlobalsVars():
    def __init__(self, iter_id=0):
        self.ts = time.time()
        self.iter_id = iter_id

        self.source = 'Pre'
        self.target = '1hr'
        
        self.current_file_path = os.path.abspath(os.path.dirname(__file__))
        print('Iter = {}'.format(self.iter_id))
        if self.iter_id < 10:
            self.filename =  os.path.join(self.current_file_path,'../EnGen_train_iterations/training_data/iter_0{0:}/Func_Pheno_45k_scaled_with_Pre_1hr_tps_source_{1:}_target_{2:}_matched.csv'.format(self.iter_id, self.source, self.target))
            self.folder_name = 'iter_0{0:}'.format(self.iter_id)
        else:
            self.filename =  os.path.join(self.current_file_path,'../EnGen_train_iterations/training_data/iter_{0:}/Func_Pheno_45k_scaled_with_Pre_1hr_tps_source_{1:}_target_{2:}_matched.csv'.format(self.iter_id, self.source, self.target))
            self.folder_name = 'iter_{0:}'.format(self.iter_id)
        
        self.dir_path_main = os.path.join(self.current_file_path,'../EnGenVAE_train_iterations/engen_vae_output/'+self.folder_name)
        self.dir_path_csv = self.dir_path_main+'/csv/'
        self.dir_path_ckpt = self.dir_path_main+'/saved_model/'
        
        os.makedirs(self.dir_path_csv, exist_ok=False)
        os.makedirs(self.dir_path_ckpt, exist_ok=False)

        self.df_gt = pd.read_csv(self.filename, memory_map=True)
        self.df_gt.drop('patient_id', axis=1, inplace=True)
        self.df_source = self.df_gt.iloc[:,:37]
        self.df_target = self.df_gt.iloc[:,37:]
        
        self.df_source = self.df_source.sample(frac=1, random_state=42)
        self.df_source.reset_index(drop=True, inplace=True)
        self.df_target = self.df_target.sample(frac=1, random_state=42)
        self.df_target.reset_index(drop=True, inplace=True)
        self.df_gt.reset_index(drop=True, inplace=True)

    
        
   




class cytofDataset(Dataset):
  

    def __init__(self, globals_vars, transform=None):

        # transform (callable, optional): Optional transform to be applied on a sample.

        self.cytof_df = pd.read_csv(globals_vars.filename, memory_map=True)
        self.cytof_df = self.cytof_df.sample(frac=1, random_state=42)
        self.cytof_df.drop('patient_id', axis=1, inplace=True)
        self.transform = transform

    def __len__(self):
        return self.cytof_df.shape[0]

    def __getitem__(self, idx):
     
        one_row_source = self.cytof_df.iloc[idx,:37].values.astype(float)
        one_row_target = self.cytof_df.iloc[idx,37:].values.astype(float)
        sample = {'row_source': one_row_source, 'row_target': one_row_target}

        if self.transform:
            sample = self.transform(sample)
        
        return sample
