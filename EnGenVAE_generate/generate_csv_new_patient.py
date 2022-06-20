import os
import sys
import numpy as np
import torch
import argparse
import pandas as pd
import pickle
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from collections import defaultdict
import torch.nn as nn
from torch.autograd import Variable
current_file_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(current_file_path, '../EnGen_model/'))  
from models import EnGenVAE



class GenerateEnGenVAE(object):
    def __init__(self, model_path, args_path, csv_path, test_patient_ids, iter_id, p, current_file_path, random_state=42):

        
        self.random_state = random_state
        self.test_patient_ids = test_patient_ids
        self.iter_id = iter_id
        self.model_path = os.path.join(current_file_path,model_path)
        self.args_path = os.path.join(current_file_path,args_path)
        self.csv_path = os.path.join(current_file_path,csv_path)
        self.p = p
        self.current_file_path = current_file_path
        os.makedirs(self.csv_path, exist_ok=True)
        self.model_args = self.get_model_arguments()
        self.device = torch.device('cuda:{}'.format(self.model_args['GPU_ID']) if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()

    def get_model_arguments(self):
        with open(self.args_path, 'r') as f:
            lines = f.readlines()
        args = {}
        for l in lines:
            l = l.replace('\n','')
            args.update({l.split('=')[0]:l.split('=')[1]})
        
        return args

    # def loss_fn(self, gen_x, x_1hr):
     
    #     mse_loss = nn.MSELoss(reduction='mean')
    #     return mse_loss(gen_x, x_1hr)
       

    def load_model(self):

        def parse_str_to_list(input_str):
            input_str = input_str.replace('[', '')
            input_str = input_str.replace(']', '')
            input_str = input_str.replace(' ', '')
            return [int(i) for i in input_str.split(',')]
       
        engen_vae = EnGenVAE(
            encoder_layer_sizes=parse_str_to_list(self.model_args['encoder_layer_sizes']),
            latent_size=int(self.model_args['latent_size']),
            decoder_layer_sizes=parse_str_to_list(self.model_args['decoder_layer_sizes']),
            device=self.device)
        print('Loading the best engen_vae model..')
        engen_vae.load_ckpt(self.model_path)

        return engen_vae

    def arcsinh_transformation(self, x):
        a = 0
        b = 1/5
        c = 0
        return np.arcsinh(a+b*x)+c

    
    def generate_csv(self):
        
        if self.iter_id < 10:
            self.iter_folder = 'iter_0{}'.format(iter_id)
        else:
            self.iter_folder = 'iter_{}'.format(iter_id)
        
        for i in range(27): # generate 27 samples
            Tensor = torch.cuda.FloatTensor if self.device=='cuda:{}'.format(self.model_args['GPU_ID']) else torch.FloatTensor
            self.model.eval()
    
            self.gen_x = self.model.inference(int(self.model_args['GPU_ID']), n=20*1000, using_cuda=torch.cuda.is_available(), random_state=42+i)
            self.gen_x = np.array(torch.Tensor.cpu(self.gen_x).detach())
            print(self.gen_x)

            markers = ['149Sm_CREB','150Nd_STAT5','151Eu_p38','153Eu_STAT1','154Sm_STAT3','155Gd_S6','159Tb_MAPKAPK2','164Dy_IkB','166Er_NFkB','167Er_ERK','168Er_pSTAT6','113In_CD235ab_CD61',
            '115In_CD45','143Nd_CD45RA','139La_CD66','141Pr_CD7','142Nd_CD19','144Nd_CD11b','145Nd_CD4','146Nd_CD8a','147Sm_CD11c','148Nd_CD123','156Gd_CD24','157Gd_CD161','158Gd_CD33',
            '165Ho_CD16','169Tm_CD25','170Er_CD3','171Yb_CD27','172Yb_CD15','173Yb_CCR2','175Lu_CD14','176Lu_CD56','160Gd_Tbet','162Dy_FoxP3','152Sm_TCRgd','174Yb_HLADR']
            
            df_gen_x = pd.DataFrame(data=self.gen_x, columns=markers)
            print(df_gen_x)
            scaler = self.p['AE_scaler']
            df_gen_x.to_csv(self.csv_path+'generated_{0:}_scaled_{1:}.csv'.format(self.iter_folder,i), header=True, index=False)
            df_gen_x.iloc[:,:] = scaler.inverse_transform(df_gen_x.iloc[:,:].values)
            df_gen_x.to_csv(self.csv_path+'generated_{0:}_{1:}.csv'.format(self.iter_folder,i), header=True, index=False)
        
        

if __name__ == "__main__":
  
    current_file_path = os.path.abspath(os.path.dirname(__file__))

    all_patient_ids = pickle.load(open(os.path.join(current_file_path,'../EnGen_train_iterations/training_data/all_patient_ids.pkl'), 'rb'))['all_patient_ids']

    n_iterations = 30 # iterate over the models trained
    
    for iter_id in range(n_iterations):
    
        if iter_id < 10:
            iter_folder = 'iter_0{}'.format(iter_id)
        else:
            iter_folder = 'iter_{}'.format(iter_id)
        path = '../EnGenVAE_train_iterations/engen_vae_output_undercomplete/{}/'.format(iter_folder)
        
        p = pickle.load(open(os.path.join(current_file_path,'../EnGen_train_iterations/training_data/{}/scaler_kmeans_25.pkl'.format(iter_folder)), 'rb'))
       
        AE_train_ids = p['AE_train_ids']
      
        test_patient_ids = [x for x in all_patient_ids if x not in AE_train_ids]
        
        model_path = path+'/saved_model/best_model_engen_vae.pth'
        args_path = path+'/commandline_args.txt'
        csv_path = path+'/generated/'
        
        generate_engen_vae = GenerateEnGenVAE(model_path, args_path, csv_path, test_patient_ids, iter_id, p, current_file_path)
        generate_engen_vae.generate_csv()
    
#   