import sys
import os
import glob
import pandas as pd 
import numpy as np 
import random
import re 
import seaborn as sns 
from matplotlib import pyplot as plt
from matplotlib import cm
import itertools
import pickle
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment






def arcsinh_transformation(x):
    a = 0
    b = 1/5
    c = 0
    return np.arcsinh(a+b*x)+c


def make_iter(save_path, data_filepath, timepoints, n_iteration=30):
    'sample one-thrid from each timepoint and learn scaler and kmeans'

    for iter_id in range(n_iteration):

        print('Iter = {}'.format(iter_id))
        random.seed(42+int(iter_id))
        all_patient_ids = pickle.load(open(save_path+'all_patient_ids.pkl', 'rb'))['all_patient_ids']
        AE_train_ids = random.sample(all_patient_ids,len(all_patient_ids)//3)
       
        if iter_id < 10:
            folder_path = save_path+'/iter_0{0:}/'.format(iter_id)
        else:
            folder_path = save_path+'/iter_{0:}/'.format(iter_id)
        os.makedirs(folder_path, exist_ok=False)
        txt_filename = folder_path+'summary.txt' 
        with open(txt_filename, 'w') as filetowrite:
            filetowrite.writelines('EnGen_train_ids = {}\n'.format('_'.join(AE_train_ids)))
            filetowrite.close()


        df_all_patients_train = pd.DataFrame()
        for tp in timepoints:
            for train_id in AE_train_ids:
                print(data_filepath+'{0:}/Func_Pheno_20k_{1:}_A_{0:}.csv'.format(tp, train_id))
                df_train_id = pd.read_csv(data_filepath+'{0:}/Func_Pheno_20k_{1:}_A_{0:}.csv'.format(tp, train_id))
                df_train_id = df_train_id.applymap(arcsinh_transformation)
                df_train_id['patient_id'] = train_id
                df_train_id['timepoint'] = tp
                df_all_patients_train = pd.concat([df_all_patients_train, df_train_id])

        df_all_patients_train = df_all_patients_train.sample(frac=1, random_state=42)
        df_all_patients_train.reset_index(inplace=True, drop=True)
        scaler = StandardScaler()
        print('fitting scaler')
        scaler.fit(df_all_patients_train.iloc[:,:-2].values)
        df_all_patients_train.iloc[:,:-2] = scaler.transform(df_all_patients_train.iloc[:,:-2].values)
        p = {}
        p['AE_train_ids'] = AE_train_ids
        p['AE_scaler'] = scaler
        print('fitting kmeans')
        n_clusters = 25
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=10000).fit(df_all_patients_train.iloc[:,:-2])
        p['AE_kmeans_{}'.format(n_clusters)] = kmeans
        pickle.dump(p, open(folder_path+'scaler_kmeans_{0:}.pkl'.format(n_clusters), 'wb'))
        df_all_patients_train.to_csv(folder_path+'Func_Pheno_20k_scaled_with_{}_tps.csv'.format('_'.join(timepoints)), header=True, index=False)
           
                
                


def compute_cost_matrix(save_path, timepoints, iter_id):
    # compute the pairwise cost matrix storing the edge weights on the bipartite matching graph
    print('Iter = {}'.format(iter_id))
    if iter_id < 10:
        folder_path = save_path+'/iter_0{0:}/'.format(iter_id)
    else:
        folder_path = save_path+'/iter_{0:}/'.format(iter_id)
    
    filename = folder_path+'Func_Pheno_20k_scaled_with_{}_tps.csv'.format('_'.join(timepoints))
    df_gt = pd.read_csv(filename, memory_map=True)
    df_source = df_gt[df_gt['timepoint']=='Pre']
    df_target = df_gt[df_gt['timepoint']=='1hr']

    AE_train_ids = pickle.load(open(folder_path+'scaler_kmeans_25.pkl', 'rb'))['AE_train_ids']
    for train_id in AE_train_ids:
        print('train_id = {}'.format(train_id))
        df_source_p = df_source[df_source['patient_id']==train_id]
        df_source_p = df_source_p.sample(n=5000, random_state=42) # reduced to 5k for speed
        df_source_p.reset_index(drop=True, inplace=True)
        df_target_p = df_target[df_target['patient_id']==train_id]
        df_target_p = df_target_p.sample(n=5000, random_state=42)
        df_target_p.reset_index(drop=True, inplace=True)
        cost = distance.cdist(df_source_p.iloc[:,:-2], df_target_p.iloc[:,:-2], metric='euclidean')
        # print(cost)
        df_cost = pd.DataFrame(data=cost)
        df_cost.to_csv(folder_path+'cost_matrix_5k_source_Pre_target_1hr_scaled_train_id_{}.csv'.format(train_id))
        # print(df_cost)


def matching(iter_id, save_path, timepoints):
    # optimize for the perfect matching

    source = 'Pre'
    target = '1hr'

    print('Iter= {}'.format(iter_id))
    if iter_id < 10:
        folder_path = save_path+'/iter_0{0:}/'.format(iter_id)
    else:
        folder_path = save_path+'/iter_{0:}/'.format(iter_id)
    
    filename = folder_path+'Func_Pheno_20k_scaled_with_{}_tps.csv'.format('_'.join(timepoints))
    df_gt = pd.read_csv(filename, memory_map=True)
    df_source = df_gt[df_gt['timepoint']=='Pre']
    df_target = df_gt[df_gt['timepoint']=='1hr']
    AE_train_ids = pickle.load(open(folder_path+'scaler_kmeans_25.pkl', 'rb'))['AE_train_ids']
    df_source_target_matched = pd.DataFrame()
    for train_id in AE_train_ids:
        cost = pd.read_csv(folder_path+'cost_matrix_5k_source_Pre_target_1hr_scaled_train_id_{}.csv'.format(train_id), index_col=0) # the first col is index
        print('finished loading csv train id = {}'.format(train_id))
        cost = np.array(cost.values)
        row_ind, col_ind = linear_sum_assignment(cost)
            
        df_source_p = df_source[df_source['patient_id']==train_id]
        df_source_p = df_source_p.sample(n=5000, random_state=42)
        df_source_p.reset_index(drop=True, inplace=True)
        df_target_p = df_target[df_target['patient_id']==train_id]
        df_target_p = df_target_p.sample(n=5000, random_state=42)
        df_target_p.reset_index(drop=True, inplace=True)
        df_source_p = df_source_p.drop('timepoint', axis=1)
        df_target_p = df_target_p.drop('timepoint', axis=1)
        df_source_p = df_source_p.drop('patient_id', axis=1)
        df_target_p = df_target_p.drop('patient_id', axis=1)

        df_source_p.columns = [col+'_{}'.format(source) for col in df_source_p.columns.values]
        df_target_p.columns = [col+'_{}'.format(target) for col in df_target_p.columns.values]
        
        df_target_p = df_target_p.reindex(col_ind)
        df_target_p.reset_index(drop=True, inplace=True)

        df_p_source_target_matched = pd.concat([df_source_p, df_target_p], axis=1)
        df_p_source_target_matched['patient_id'] = train_id
        df_source_target_matched = pd.concat([df_source_target_matched, df_p_source_target_matched], axis=0)

    print('finished. Writing to csv..')
    df_source_target_matched.to_csv(folder_path+'Func_Pheno_45k_scaled_with_{}_tps_source_{}_target_{}_matched.csv'.format('_'.join(timepoints), source, target), index=False, header=True)
      

if __name__ == "__main__":
    
    current_file_path = os.path.abspath(os.path.dirname(__file__))
    save_folder_path = os.path.join(current_file_path,'./EnGen_train_iterations/training_data/')
    os.makedirs(save_folder_path, exist_ok=True)
    data_filepath = os.path.join(current_file_path,'./data/preprocessed/')
    timepoints = ['Pre', '1hr'] 

    ################
    # all_patient_ids_Pre = []
    # all_patient_ids_1hr = []
    # all_patient_ids_Pre.extend(glob.glob(data_filepath+'Pre/Func_Pheno_20k_*_A_Pre.csv'))
    # all_patient_ids_1hr.extend(glob.glob(data_filepath+'1hr/Func_Pheno_20k_*_A_1hr.csv'))
    # all_patient_ids_Pre = [folder.split('/')[-1].split('.csv')[0].split('_')[3] for folder in all_patient_ids_Pre]
    # all_patient_ids_1hr = [folder.split('/')[-1].split('.csv')[0].split('_')[3] for folder in all_patient_ids_1hr]
    # all_patient_ids = [x for x in all_patient_ids_Pre if x in all_patient_ids_1hr]
    # p = {}
    # p['all_patient_ids'] = all_patient_ids
    # pickle.dump(p, open(save_folder_path+'all_patient_ids.pkl', 'wb')) # uncomment this block if the pickle file doesn't exist
    ##############

    n_iterations = 30
    make_iter(save_folder_path, data_filepath, timepoints, n_iterations) #slicing training patients for each iteration
    
    for iter_id in range(n_iterations):
        compute_cost_matrix(save_folder_path, timepoints, iter_id)
        matching(iter_id, save_folder_path, timepoints)



    
    
