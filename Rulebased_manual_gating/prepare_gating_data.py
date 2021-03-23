import sys
import os
import pandas as pd 
import numpy as np 
import random
import pickle


def inv_arcsinh_transformation(x):
    a = 0
    b = 1/5
    c = 0

    return (np.sinh(x-c)-a)/b

def all_gt(timepoints, current_file_path):

    #concat gt csv at timepoints

    all_patients = pickle.load(open(os.path.join(current_file_path,'../EnGen_train_iterations/training_data/all_patient_ids.pkl'), 'rb'))['all_patient_ids']

    column_labels = ['149Sm_CREB', '150Nd_STAT5', '151Eu_p38', '153Eu_STAT1', '154Sm_STAT3', '155Gd_S6', '159Tb_MAPKAPK2', '164Dy_IkB', '166Er_NFkB', '167Er_ERK', '168Er_pSTAT6', '113In_CD235ab_CD61', '115In_CD45', '143Nd_CD45RA', '139La_CD66',
                    '141Pr_CD7', '142Nd_CD19', '144Nd_CD11b', '145Nd_CD4', '146Nd_CD8a', '147Sm_CD11c', '148Nd_CD123', '156Gd_CD24', '157Gd_CD161', '158Gd_CD33', '165Ho_CD16', '169Tm_CD25', '170Er_CD3', '171Yb_CD27', '172Yb_CD15',
                    '173Yb_CCR2', '175Lu_CD14',	'176Lu_CD56', '160Gd_Tbet', '162Dy_FoxP3', '152Sm_TCRgd', '174Yb_HLADR']
    df_all_gt = pd.DataFrame(columns=column_labels)
    for patient_id in all_patients:
        for timepoint in timepoints:
            
            df_train_id = pd.read_csv(os.path.join(current_file_path,'../data/preprocessed/{0:}/Func_Pheno_20k_{1:}_A_{0:}.csv'.format(timepoint, patient_id)))
            assert len(column_labels)==len(df_train_id.columns.values), 'Columns do not match!'
            df_train_id.columns = column_labels
            df_train_id['patient_id'] = patient_id
            df_train_id['timepoint'] = timepoint
            df_all_gt = pd.concat([df_all_gt, df_train_id], axis=0) 
            
    df_all_gt = df_all_gt.sample(frac=1, random_state=42)
    df_all_gt.reset_index(drop=True, inplace=True)
    os.makedirs(os.path.join(current_file_path,'../data/gates/'), exist_ok=True)
    df_all_gt.to_csv(os.path.join(current_file_path,'../data/gates/gt_Pre_1hr_all_patients.csv'), index=False, header=True)
    
    

    

        

def all_generated_train(current_file_path):

    #concat gen csv trained on same patient using all models
    
    all_patients = pickle.load(open(os.path.join(current_file_path,'../EnGen_train_iterations/training_data/all_patient_ids.pkl'), 'rb'))['all_patient_ids']

    column_labels = ['149Sm_CREB', '150Nd_STAT5', '151Eu_p38', '153Eu_STAT1', '154Sm_STAT3', '155Gd_S6', '159Tb_MAPKAPK2', '164Dy_IkB', '166Er_NFkB', '167Er_ERK', '168Er_pSTAT6', '113In_CD235ab_CD61', '115In_CD45', '143Nd_CD45RA', '139La_CD66',
                    '141Pr_CD7', '142Nd_CD19', '144Nd_CD11b', '145Nd_CD4', '146Nd_CD8a', '147Sm_CD11c', '148Nd_CD123', '156Gd_CD24', '157Gd_CD161', '158Gd_CD33', '165Ho_CD16', '169Tm_CD25', '170Er_CD3', '171Yb_CD27', '172Yb_CD15',
                    '173Yb_CCR2', '175Lu_CD14',	'176Lu_CD56', '160Gd_Tbet', '162Dy_FoxP3', '152Sm_TCRgd', '174Yb_HLADR']

    for source_id in all_patients:
        
        df_all_gen = pd.DataFrame(columns=column_labels)
        print('-----------------source_id = {}'.format(source_id))
        for iter_id in range(30):
            print('----iter_id = {}'.format(iter_id))
            # iter_id = random.randint(0,29) # randomly pick a model
            if iter_id < 10:
                iter_id = 'iter_0{}'.format(iter_id)
            else:
                iter_id = 'iter_{}'.format(iter_id)
            
            AE_train_ids = pickle.load(open(os.path.join(current_file_path,'../EnGen_train_iterations/training_data/{}/scaler_kmeans_25.pkl'.format(iter_id)), 'rb'))['AE_train_ids']
        
            if source_id not in AE_train_ids:
                print('**Not in train')
                continue
            df_sample_gen = pd.read_csv(os.path.join(current_file_path,'../EnGen_train_iterations/engen_output/{0:}/generated_train/generated_{1:}.csv'.format(iter_id, source_id)))
            df_sample_gen = df_sample_gen.applymap(inv_arcsinh_transformation)
            df_sample_gen[df_sample_gen < 0] = 0
            assert len(column_labels)==len(df_sample_gen.columns.values), 'Columns do not match!'
            df_sample_gen.columns = column_labels
            df_sample_gen.reset_index(drop=True, inplace=True)
            
            df_all_gen = pd.concat([df_all_gen, df_sample_gen], axis=0)
            
        
        df_all_gen.reset_index(drop=True, inplace=True)
        os.makedirs(os.path.join(current_file_path,'../data/gates/gen_train/'), exist_ok=True)
        df_all_gen.to_csv(os.path.join(current_file_path,'../data/gates/gen_train/gen_train_{}_all_models_inv_scaled_inv_arcsinh.csv'.format(source_id)), index=False, header=True)

def all_generated_test(current_file_path):

    #concat gen csv not trained on those patient using all models
    
    all_patients = pickle.load(open(os.path.join(current_file_path,'../EnGen_train_iterations/training_data/all_patient_ids.pkl'), 'rb'))['all_patient_ids']

    column_labels = ['149Sm_CREB', '150Nd_STAT5', '151Eu_p38', '153Eu_STAT1', '154Sm_STAT3', '155Gd_S6', '159Tb_MAPKAPK2', '164Dy_IkB', '166Er_NFkB', '167Er_ERK', '168Er_pSTAT6', '113In_CD235ab_CD61', '115In_CD45', '143Nd_CD45RA', '139La_CD66',
                    '141Pr_CD7', '142Nd_CD19', '144Nd_CD11b', '145Nd_CD4', '146Nd_CD8a', '147Sm_CD11c', '148Nd_CD123', '156Gd_CD24', '157Gd_CD161', '158Gd_CD33', '165Ho_CD16', '169Tm_CD25', '170Er_CD3', '171Yb_CD27', '172Yb_CD15',
                    '173Yb_CCR2', '175Lu_CD14',	'176Lu_CD56', '160Gd_Tbet', '162Dy_FoxP3', '152Sm_TCRgd', '174Yb_HLADR']

    for test_id in all_patients:
        
        df_all_gen = pd.DataFrame(columns=column_labels)
        print('-------test_id = {}'.format(test_id))
        for iter_id in range(30):
            print('---iter_id = {}'.format(iter_id))

            if iter_id < 10:
                iter_id = 'iter_0{}'.format(iter_id)
            else:
                iter_id = 'iter_{}'.format(iter_id)
            
            AE_train_ids = pickle.load(open(os.path.join(current_file_path,'../EnGen_train_iterations/training_data/{}/scaler_kmeans_25.pkl'.format(iter_id)), 'rb'))['AE_train_ids']
            if test_id in AE_train_ids:
                print('**in train')
                continue
            df_sample_gen = pd.read_csv(os.path.join(current_file_path,'../EnGen_train_iterations/engen_output/{0:}/generated/generated_{1:}.csv'.format(iter_id, test_id)))
            df_sample_gen = df_sample_gen.applymap(inv_arcsinh_transformation)
            df_sample_gen[df_sample_gen < 0] = 0
            assert len(column_labels)==len(df_sample_gen.columns.values), 'Columns do not match!'
            df_sample_gen.columns = column_labels
            df_sample_gen.reset_index(drop=True, inplace=True)
            
            df_all_gen = pd.concat([df_all_gen, df_sample_gen], axis=0)
        
        df_all_gen.reset_index(drop=True, inplace=True)
        os.makedirs(os.path.join(current_file_path,'../data/gates/gen_test/'), exist_ok=True)
        df_all_gen.to_csv(os.path.join(current_file_path,'../data/gates/gen_test/gen_test_{}_all_models_inv_scaled_inv_arcsinh.csv'.format(test_id)), index=False, header=True)

        


if __name__ == "__main__":
    
    current_file_path = os.path.abspath(os.path.dirname(__file__))
    timepoints = ['Pre', '1hr']
    # all_gt(timepoints, current_file_path)

    # all_generated_train(current_file_path) # train_generated samples by all models for one patient
    all_generated_test(current_file_path) # test_generated samples by all models for one patient 






    
    
