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
from models import EnGen



class GenerateEnGen(object):
    def __init__(self, model_path, args_path, csv_path, test_patient_ids, source, target, iter_id, p, current_file_path, random_state=42):

        
        self.random_state = random_state
        self.test_patient_ids = test_patient_ids
        self.iter_id = iter_id
        self.model_path = os.path.join(current_file_path,model_path)
        self.args_path = os.path.join(current_file_path,args_path)
        self.csv_path = os.path.join(current_file_path,csv_path)
        self.p = p
        self.source = source
        self.target = target
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

    def loss_fn(self, gen_x, x_1hr):
     
        mse_loss = nn.MSELoss(reduction='mean')
        return mse_loss(gen_x, x_1hr)
       

    def load_model(self):

        def parse_str_to_list(input_str):
            input_str = input_str.replace('[', '')
            input_str = input_str.replace(']', '')
            input_str = input_str.replace(' ', '')
            return [int(i) for i in input_str.split(',')]
       
        engen = EnGen(
            encoder_layer_sizes=parse_str_to_list(self.model_args['encoder_layer_sizes']),
            latent_size=int(self.model_args['latent_size']),
            decoder_layer_sizes=parse_str_to_list(self.model_args['decoder_layer_sizes']),
            device=self.device)
        print('Loading the best engen model..')
        engen.load_ckpt(self.model_path)

        return engen

    def arcsinh_transformation(self, x):
        a = 0
        b = 1/5
        c = 0
        return np.arcsinh(a+b*x)+c

    
    def generate_csv(self):
        
        for test_id in self.test_patient_ids:
            print('generating for test patient {}'.format(test_id))
            df_source = pd.read_csv(os.path.join(self.current_file_path,'../data/preprocessed/Pre/Func_Pheno_20k_{0:}_A_Pre.csv'.format(test_id)))
            # print(df_source)
            df_source = df_source.applymap(self.arcsinh_transformation)
            scaler = self.p['AE_scaler']
            df_source.iloc[:,:] = scaler.transform(df_source.iloc[:,:].values)
            
            df_source = df_source.sample(frac=1, random_state=42)
            df_source.reset_index(drop=True, inplace=True)
            Tensor = torch.cuda.FloatTensor if self.device=='cuda:{}'.format(self.model_args['GPU_ID']) else torch.FloatTensor

            self.model.eval()
            x_source = torch.from_numpy(df_source.values)
           
            x_source = Variable(x_source.type(Tensor)).cuda(int(self.model_args['GPU_ID']))
            
            gen_target, z_source = self.model(x_source)

           
            df_gen_target = pd.DataFrame(data=np.array(torch.Tensor.cpu(gen_target).detach()), columns=df_source.columns.values)


            df_gen_target.to_csv(self.csv_path+'generated_{}_scaled.csv'.format(test_id), header=True, index=False)
            df_gen_target.iloc[:,:] = scaler.inverse_transform(df_gen_target.iloc[:,:].values)
            df_gen_target.to_csv(self.csv_path+'generated_{}.csv'.format(test_id), header=True, index=False)
         
        

if __name__ == "__main__":
  
    current_file_path = os.path.abspath(os.path.dirname(__file__))
    source = 'Pre'
    target = '1hr'    

    all_patient_ids = pickle.load(open(os.path.join(current_file_path,'../EnGen_train_iterations/training_data/all_patient_ids.pkl'), 'rb'))['all_patient_ids']

    
    # for iter_id in range(30):
    for iter_id in [0]:
    
        if iter_id < 10:
            iter_folder = 'iter_0{}'.format(iter_id)
        else:
            iter_folder = 'iter_{}'.format(iter_id)
        path = '../EnGen_train_iterations/engen_output/{}/'.format(iter_folder)
        
        p = pickle.load(open(os.path.join(current_file_path,'../EnGen_train_iterations/training_data/{}/scaler_kmeans_25.pkl'.format(iter_folder)), 'rb'))
       
        AE_train_ids = p['AE_train_ids']
      
        test_patient_ids = [x for x in all_patient_ids if x not in AE_train_ids]
        
        model_path = path+'/saved_model/best_model_engen.pth'
        args_path = path+'/commandline_args.txt'
        csv_path = path+'/generated/'
        
        generate_engen = GenerateEnGen(model_path, args_path, csv_path, test_patient_ids, source, target, iter_id, p, current_file_path)
        generate_engen.generate_csv()
    
#   