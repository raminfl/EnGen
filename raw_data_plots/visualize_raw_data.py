import sys
import os
import pandas as pd 
import numpy as np 
import random
import re 
from matplotlib import pyplot as plt
from matplotlib import cm
import pickle
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from umap import UMAP
import plotly.graph_objects as go


def one_gen_all_models_train_PCA(current_file_path):
    #plot PCA for generated samples from a patient using all models that had the patient as train
    random.seed(42)
    all_patients = pickle.load(open(os.path.join(current_file_path,'../EnGen_train_iterations/training_data/all_patient_ids.pkl'), 'rb'))['all_patient_ids']
    column_labels = ['149Sm_CREB', '150Nd_STAT5', '151Eu_p38', '153Eu_STAT1', '154Sm_STAT3', '155Gd_S6', '159Tb_MAPKAPK2', '164Dy_IkB', '166Er_NFkB', '167Er_ERK', '168Er_pSTAT6', '113In_CD235ab_CD61', '115In_CD45', '143Nd_CD45RA', '139La_CD66',
                    '141Pr_CD7', '142Nd_CD19', '144Nd_CD11b', '145Nd_CD4', '146Nd_CD8a', '147Sm_CD11c', '148Nd_CD123', '156Gd_CD24', '157Gd_CD161', '158Gd_CD33', '165Ho_CD16', '169Tm_CD25', '170Er_CD3', '171Yb_CD27', '172Yb_CD15',
                    '173Yb_CCR2', '175Lu_CD14',	'176Lu_CD56', '160Gd_Tbet', '162Dy_FoxP3', '152Sm_TCRgd', '174Yb_HLADR']
    df_all_gt_Pre_1hr = pd.read_csv(os.path.join(current_file_path,'../data/gates/gt_Pre_1hr_all_patients.csv'))
    df_all_gt_Pre =  df_all_gt_Pre_1hr[df_all_gt_Pre_1hr['timepoint']=='Pre']
    df_all_gt_1hr =  df_all_gt_Pre_1hr[df_all_gt_Pre_1hr['timepoint']=='1hr']
    df_all_gt_Pre_1hr.drop(['timepoint', 'patient_id'], axis=1, inplace=True)
    df_all_gt_Pre_1hr = df_all_gt_Pre_1hr.sample(n=20000, random_state=42)
    df_all_gt_Pre_1hr.reset_index(drop=True, inplace=True)
    scaler = pickle.load(open(os.path.join(current_file_path,'../data/gates/All_patients_gt_Pre_1hr_scaler_pca.pkl'), 'rb'))['all_gt_Pre_1hr_scaler']
    pca = pickle.load(open(os.path.join(current_file_path,'../data/gates/All_patients_gt_Pre_1hr_scaler_pca.pkl'), 'rb'))['all_gt_Pre_1hr_pca']
    df_all_gt_Pre_1hr.loc[:,:] = scaler.transform(df_all_gt_Pre_1hr.values)
    df_pca_Pre_1hr = pd.DataFrame(data=pca.transform(df_all_gt_Pre_1hr), columns=['PC1', 'PC2'])


    plot_patient = random.sample(all_patients, 1)[0] # select one patient randomly for plotting in Fig2 hexbin
    print('plot_patient = {}'.format(plot_patient))
    # for patient_id in all_patients: # generate the plots for all patients
    for patient_id in [plot_patient]: # only generate the plot_patient
    
        df_all_gt_Pre_p =  df_all_gt_Pre[df_all_gt_Pre['patient_id']==patient_id]
        df_all_gt_1hr_p =  df_all_gt_1hr[df_all_gt_1hr['patient_id']==patient_id]
        
        df_all_gt_Pre_p.drop(['timepoint', 'patient_id'], axis=1, inplace=True)
        df_all_gt_1hr_p.drop(['timepoint', 'patient_id'], axis=1, inplace=True)  
        
        df_all_gt_Pre_p = df_all_gt_Pre_p.sample(frac=1, random_state=42)
        df_all_gt_1hr_p = df_all_gt_1hr_p.sample(frac=1, random_state=42)
    
        df_all_gt_Pre_p.reset_index(drop=True, inplace=True)
        df_all_gt_1hr_p.reset_index(drop=True, inplace=True)

        df_sample_gen = pd.read_csv(os.path.join(current_file_path,'../data/gates/gen_train/gen_train_{}_all_models_inv_scaled_inv_arcsinh.csv'.format(patient_id)))
        assert len(column_labels)==len(df_sample_gen.columns.values), 'Columns do not match!'
        df_sample_gen.columns = column_labels

        df_sample_gen = df_sample_gen.sample(n=20000, random_state=42)
        df_sample_gen.reset_index(drop=True, inplace=True)
        df_all_gt_Pre_p.loc[:,:] = scaler.transform(df_all_gt_Pre_p.values)
        df_all_gt_1hr_p.loc[:,:] = scaler.transform(df_all_gt_1hr_p.values)
        df_sample_gen.loc[:,:] = scaler.transform(df_sample_gen.values)
        
        #plot PCA
        df_pca_Pre = pd.DataFrame(data=pca.transform(df_all_gt_Pre_p), columns=['PC1', 'PC2'])
        df_pca_1hr = pd.DataFrame(data=pca.transform(df_all_gt_1hr_p), columns=['PC1', 'PC2'])
        df_pca_gen = pd.DataFrame(data=pca.transform(df_sample_gen), columns=['PC1', 'PC2'])
        
        fig, ax = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False, figsize=(17,4))

        ################ PCA scatter plot #########################
        df_pca_Pre.plot(kind='scatter', x='PC1', y='PC2', ax=ax[0], c='black', s=5)
        df_pca_1hr.plot(kind='scatter', x='PC1', y='PC2', ax=ax[1], c='black', s=5)
        df_pca_gen.plot(kind='scatter', x='PC1', y='PC2', ax=ax[2], c='black', s=5)
        ax[0].set_title('{} GT Pre'.format(patient_id))  
        ax[1].set_title('{} GT 1hr'.format(patient_id))  
        ax[2].set_title('{} train_GEN 1hr'.format(patient_id))
        ax[0].set_ylim((-10,12))
        ax[0].set_xlim((-10,12))
        ax[1].set_ylim((-10,12))
        ax[1].set_xlim((-10,12))
        ax[2].set_ylim((-10,12))
        ax[2].set_xlim((-10,12))
        os.makedirs(os.path.join(current_file_path,'../raw_data_plots/plots/PCA_raw_Pre_1hr/gen_train_all_models/'),exist_ok=True)
        plt.savefig(os.path.join(current_file_path,'../raw_data_plots/plots/PCA_raw_Pre_1hr/gen_train_all_models/Scatter_plot_PCA_Pre_1hr_gen_train_{}_all_models.pdf'.format(patient_id)), format='pdf', dpi=600)
        
        ################ PCA hexbin plot #########################
        fig, ax = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False, figsize=(17,4))
        hb_Pre = ax[0].hexbin(x=df_pca_Pre['PC1'], y=df_pca_Pre['PC2'], bins='log', cmap='inferno', extent=(df_pca_Pre['PC1'].min(), df_pca_Pre['PC1'].max(), df_pca_Pre['PC2'].min(), df_pca_Pre['PC2'].max()))
        hb_1hr = ax[1].hexbin(x=df_pca_1hr['PC1'], y=df_pca_1hr['PC2'], bins='log', cmap='inferno', extent=(df_pca_Pre['PC1'].min(), df_pca_Pre['PC1'].max(), df_pca_Pre['PC2'].min(), df_pca_Pre['PC2'].max()))
        hb_gen = ax[2].hexbin(x=df_pca_gen['PC1'], y=df_pca_gen['PC2'], bins='log', cmap='inferno', extent=(df_pca_Pre['PC1'].min(), df_pca_Pre['PC1'].max(), df_pca_Pre['PC2'].min(), df_pca_Pre['PC2'].max()))
        max_ploy_count = max(max(hb_Pre.get_array()),max(hb_1hr.get_array()),max(hb_gen.get_array()))
    
        plt.close()
        fig, ax = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False, figsize=(17,4))
        
        hb_Pre = ax[0].hexbin(x=df_pca_Pre['PC1'], y=df_pca_Pre['PC2'], bins='log', cmap='inferno', vmax=max_ploy_count, extent=(-6, 10, -6, 10))
        hb_1hr = ax[1].hexbin(x=df_pca_1hr['PC1'], y=df_pca_1hr['PC2'], bins='log', cmap='inferno', vmax=max_ploy_count, extent=(-6, 10, -6, 10))
        hb_gen = ax[2].hexbin(x=df_pca_gen['PC1'], y=df_pca_gen['PC2'], bins='log', cmap='inferno', vmax=max_ploy_count, extent=(-6, 10, -6, 10))
       
        ax[0].set_title('{} GT Pre'.format(patient_id))  
        ax[1].set_title('{} GT 1hr'.format(patient_id))  
        ax[2].set_title('{} train_GEN 1hr'.format(patient_id))
       
        plt.savefig(os.path.join(current_file_path,'../raw_data_plots/plots/PCA_raw_Pre_1hr/gen_train_all_models/Hexbin_plot_PCA_Pre_1hr_gen_train_{}_all_models.pdf'.format(patient_id)), format='pdf', dpi=600)
        plt.close()




def one_gen_all_models_test_PCA(current_file_path):
    
    #plot PCA for generated samples from a patient using all models that had the patient as test 
    random.seed(42)
    all_patients = pickle.load(open(os.path.join(current_file_path,'../EnGen_train_iterations/training_data/all_patient_ids.pkl'), 'rb'))['all_patient_ids']
    column_labels = ['149Sm_CREB', '150Nd_STAT5', '151Eu_p38', '153Eu_STAT1', '154Sm_STAT3', '155Gd_S6', '159Tb_MAPKAPK2', '164Dy_IkB', '166Er_NFkB', '167Er_ERK', '168Er_pSTAT6', '113In_CD235ab_CD61', '115In_CD45', '143Nd_CD45RA', '139La_CD66',
                    '141Pr_CD7', '142Nd_CD19', '144Nd_CD11b', '145Nd_CD4', '146Nd_CD8a', '147Sm_CD11c', '148Nd_CD123', '156Gd_CD24', '157Gd_CD161', '158Gd_CD33', '165Ho_CD16', '169Tm_CD25', '170Er_CD3', '171Yb_CD27', '172Yb_CD15',
                    '173Yb_CCR2', '175Lu_CD14',	'176Lu_CD56', '160Gd_Tbet', '162Dy_FoxP3', '152Sm_TCRgd', '174Yb_HLADR']
    df_all_gt_Pre_1hr = pd.read_csv(os.path.join(current_file_path,'../data/gates/gt_Pre_1hr_all_patients.csv'))
    df_all_gt_Pre =  df_all_gt_Pre_1hr[df_all_gt_Pre_1hr['timepoint']=='Pre']
    df_all_gt_1hr =  df_all_gt_Pre_1hr[df_all_gt_Pre_1hr['timepoint']=='1hr']
    df_all_gt_Pre_1hr.drop(['timepoint', 'patient_id'], axis=1, inplace=True)
    df_all_gt_Pre_1hr = df_all_gt_Pre_1hr.sample(n=20000, random_state=42)
    df_all_gt_Pre_1hr.reset_index(drop=True, inplace=True)
    scaler = pickle.load(open(os.path.join(current_file_path,'../data/gates/All_patients_gt_Pre_1hr_scaler_pca.pkl'), 'rb'))['all_gt_Pre_1hr_scaler']
    pca = pickle.load(open(os.path.join(current_file_path,'../data/gates/All_patients_gt_Pre_1hr_scaler_pca.pkl'), 'rb'))['all_gt_Pre_1hr_pca']
    df_all_gt_Pre_1hr.loc[:,:] = scaler.transform(df_all_gt_Pre_1hr.values)
    df_pca_Pre_1hr = pd.DataFrame(data=pca.transform(df_all_gt_Pre_1hr), columns=['PC1', 'PC2'])

    
    plot_patient = random.sample(all_patients, 1)[0] # select one patient randomly for plotting in Fig2 hexbin
    print('plot_patient = {}'.format(plot_patient))
    # for patient_id in all_patients: # generate the plots for all patients
    for patient_id in [plot_patient]: # only generate the plot_patient
    
        df_all_gt_Pre_p =  df_all_gt_Pre[df_all_gt_Pre['patient_id']==patient_id]
        df_all_gt_1hr_p =  df_all_gt_1hr[df_all_gt_1hr['patient_id']==patient_id]
        
        df_all_gt_Pre_p.drop(['timepoint', 'patient_id'], axis=1, inplace=True)
        df_all_gt_1hr_p.drop(['timepoint', 'patient_id'], axis=1, inplace=True)  
        
        df_all_gt_Pre_p = df_all_gt_Pre_p.sample(frac=1, random_state=42)
        df_all_gt_1hr_p = df_all_gt_1hr_p.sample(frac=1, random_state=42)
        
        df_all_gt_Pre_p.reset_index(drop=True, inplace=True)
        df_all_gt_1hr_p.reset_index(drop=True, inplace=True)

        df_sample_gen = pd.read_csv(os.path.join(current_file_path,'../data/gates/gen_test/gen_test_{}_all_models_inv_scaled_inv_arcsinh.csv'.format(patient_id)))
        assert len(column_labels)==len(df_sample_gen.columns.values), 'Columns do not match!'
        df_sample_gen.columns = column_labels

        df_sample_gen = df_sample_gen.sample(n=20000, random_state=42)
        df_sample_gen.reset_index(drop=True, inplace=True)

        df_all_gt_Pre_p.loc[:,:] = scaler.transform(df_all_gt_Pre_p.values)
        df_all_gt_1hr_p.loc[:,:] = scaler.transform(df_all_gt_1hr_p.values)
        df_sample_gen.loc[:,:] = scaler.transform(df_sample_gen.values)
        
  
        df_pca_Pre = pd.DataFrame(data=pca.transform(df_all_gt_Pre_p), columns=['PC1', 'PC2'])
        df_pca_1hr = pd.DataFrame(data=pca.transform(df_all_gt_1hr_p), columns=['PC1', 'PC2'])
        df_pca_gen = pd.DataFrame(data=pca.transform(df_sample_gen), columns=['PC1', 'PC2'])
        
        ################ PCA hexbin plot #########################
        fig, ax = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False, figsize=(17,4))
        hb_Pre = ax[0].hexbin(x=df_pca_Pre['PC1'], y=df_pca_Pre['PC2'], bins='log', cmap='inferno', extent=(df_pca_Pre['PC1'].min(), df_pca_Pre['PC1'].max(), df_pca_Pre['PC2'].min(), df_pca_Pre['PC2'].max()))
        hb_1hr = ax[1].hexbin(x=df_pca_1hr['PC1'], y=df_pca_1hr['PC2'], bins='log', cmap='inferno', extent=(df_pca_Pre['PC1'].min(), df_pca_Pre['PC1'].max(), df_pca_Pre['PC2'].min(), df_pca_Pre['PC2'].max()))
        hb_gen = ax[2].hexbin(x=df_pca_gen['PC1'], y=df_pca_gen['PC2'], bins='log', cmap='inferno', extent=(df_pca_Pre['PC1'].min(), df_pca_Pre['PC1'].max(), df_pca_Pre['PC2'].min(), df_pca_Pre['PC2'].max()))
        max_ploy_count = max(max(hb_Pre.get_array()),max(hb_1hr.get_array()),max(hb_gen.get_array()))
        plt.close()
        fig, ax = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False, figsize=(17,4))
        hb_Pre = ax[0].hexbin(x=df_pca_Pre['PC1'], y=df_pca_Pre['PC2'], bins='log', cmap='inferno', vmax=max_ploy_count, extent=(-6, 10, -6, 10))
        hb_1hr = ax[1].hexbin(x=df_pca_1hr['PC1'], y=df_pca_1hr['PC2'], bins='log', cmap='inferno', vmax=max_ploy_count, extent=(-6, 10, -6, 10))
        hb_gen = ax[2].hexbin(x=df_pca_gen['PC1'], y=df_pca_gen['PC2'], bins='log', cmap='inferno', vmax=max_ploy_count, extent=(-6, 10, -6, 10))
        
        ax[0].set_title('{} GT Pre'.format(patient_id))  
        ax[1].set_title('{} GT 1hr'.format(patient_id))  
        ax[2].set_title('{} train_GEN 1hr'.format(patient_id))
        
        os.makedirs(os.path.join(current_file_path,'../raw_data_plots/plots/PCA_raw_Pre_1hr/gen_test_all_models/'),exist_ok=True)
        plt.savefig(os.path.join(current_file_path,'../raw_data_plots/plots/PCA_raw_Pre_1hr/gen_test_all_models/Hexbin_plot_PCA_Pre_1hr_gen_test_{}_all_models.pdf'.format(patient_id)), format='pdf', dpi=600)
        plt.close()


        

def one_gen_all_models_train_and_test_UMAP(current_file_path):
    # Plot UMAP for generated samples from a patient using all models that had the patient as train
    # also plot UMAP for generated samples from a patient using all models that had the patient as test 
    random.seed(42)
    all_patients = pickle.load(open(os.path.join(current_file_path,'../EnGen_train_iterations/training_data/all_patient_ids.pkl'), 'rb'))['all_patient_ids']
    column_labels = ['149Sm_CREB', '150Nd_STAT5', '151Eu_p38', '153Eu_STAT1', '154Sm_STAT3', '155Gd_S6', '159Tb_MAPKAPK2', '164Dy_IkB', '166Er_NFkB', '167Er_ERK', '168Er_pSTAT6', '113In_CD235ab_CD61', '115In_CD45', '143Nd_CD45RA', '139La_CD66',
                    '141Pr_CD7', '142Nd_CD19', '144Nd_CD11b', '145Nd_CD4', '146Nd_CD8a', '147Sm_CD11c', '148Nd_CD123', '156Gd_CD24', '157Gd_CD161', '158Gd_CD33', '165Ho_CD16', '169Tm_CD25', '170Er_CD3', '171Yb_CD27', '172Yb_CD15',
                    '173Yb_CCR2', '175Lu_CD14',	'176Lu_CD56', '160Gd_Tbet', '162Dy_FoxP3', '152Sm_TCRgd', '174Yb_HLADR']
    df_all_gt_Pre_1hr = pd.read_csv(os.path.join(current_file_path,'../data/gates/gt_Pre_1hr_all_patients.csv'))
    df_all_gt_Pre =  df_all_gt_Pre_1hr[df_all_gt_Pre_1hr['timepoint']=='Pre']
    df_all_gt_1hr =  df_all_gt_Pre_1hr[df_all_gt_Pre_1hr['timepoint']=='1hr']
    df_all_gt_Pre_1hr.drop(['timepoint', 'patient_id'], axis=1, inplace=True)
    df_all_gt_Pre_1hr = df_all_gt_Pre_1hr.sample(n=20000, random_state=42)
    df_all_gt_Pre_1hr.reset_index(drop=True, inplace=True)
    scaler = pickle.load(open(os.path.join(current_file_path,'../data/gates/All_patients_gt_Pre_1hr_scaler_pca.pkl'), 'rb'))['all_gt_Pre_1hr_scaler']
    df_all_gt_Pre_1hr.loc[:,:] = scaler.transform(df_all_gt_Pre_1hr.values)

    my_umap = UMAP(n_neighbors=5,
                      min_dist=0.3,
                      metric='correlation').fit(df_all_gt_Pre_1hr)

    plot_patient = random.sample(all_patients, 1)[0] # select one patient randomly for plotting in Fig2 hexbin
    print('plot_patient = {}'.format(plot_patient))
    for patient_id in all_patients: # generate the plots for all patients
    # for patient_id in [plot_patient]: # only generate the plot_patient
    
        df_all_gt_Pre_p =  df_all_gt_Pre[df_all_gt_Pre['patient_id']==patient_id]
        df_all_gt_1hr_p =  df_all_gt_1hr[df_all_gt_1hr['patient_id']==patient_id]
        
        df_all_gt_Pre_p.drop(['timepoint', 'patient_id'], axis=1, inplace=True)
        df_all_gt_1hr_p.drop(['timepoint', 'patient_id'], axis=1, inplace=True)  
        
        df_all_gt_Pre_p = df_all_gt_Pre_p.sample(frac=1, random_state=42)
        df_all_gt_1hr_p = df_all_gt_1hr_p.sample(frac=1, random_state=42)
    
        df_all_gt_Pre_p.reset_index(drop=True, inplace=True)
        df_all_gt_1hr_p.reset_index(drop=True, inplace=True)
        df_all_gt_Pre_p.loc[:,:] = scaler.transform(df_all_gt_Pre_p.values)
        df_all_gt_1hr_p.loc[:,:] = scaler.transform(df_all_gt_1hr_p.values)

        df_sample_gen = pd.read_csv(os.path.join(current_file_path,'../data/gates/gen_train/gen_train_{}_all_models_inv_scaled_inv_arcsinh.csv'.format(patient_id)))
        assert len(column_labels)==len(df_sample_gen.columns.values), 'Columns do not match!'
        df_sample_gen.columns = column_labels

        df_sample_gen = df_sample_gen.sample(n=20000, random_state=42)
        df_sample_gen.reset_index(drop=True, inplace=True)
        df_sample_gen.loc[:,:] = scaler.transform(df_sample_gen.values)

        df_sample_gen_test = pd.read_csv(os.path.join(current_file_path,'../data/gates/gen_test/gen_test_{}_all_models_inv_scaled_inv_arcsinh.csv'.format(patient_id)))
        assert len(column_labels)==len(df_sample_gen_test.columns.values), 'Columns do not match!'
        df_sample_gen_test.columns = column_labels

        df_sample_gen_test = df_sample_gen_test.sample(n=20000, random_state=42)
        df_sample_gen_test.reset_index(drop=True, inplace=True)
        df_sample_gen_test.loc[:,:] = scaler.transform(df_sample_gen_test.values)

        

        #plot UMAP
        df_umap_Pre = pd.DataFrame(data=my_umap.transform(df_all_gt_Pre_p), columns=['UMAP1', 'UMAP2'])
        df_umap_1hr = pd.DataFrame(data=my_umap.transform(df_all_gt_1hr_p), columns=['UMAP1', 'UMAP2'])
        df_umap_gen = pd.DataFrame(data=my_umap.transform(df_sample_gen), columns=['UMAP1', 'UMAP2'])
        df_umap_gen_test = pd.DataFrame(data=my_umap.transform(df_sample_gen_test), columns=['UMAP1', 'UMAP2'])
        
        ################ UMAP hexbin plot #########################
        fig, ax = plt.subplots(nrows=1, ncols=4, sharex=False, sharey=False, figsize=(22,4))
        hb_Pre = ax[0].hexbin(x=df_umap_Pre['UMAP1'], y=df_umap_Pre['UMAP2'], bins='log', cmap='inferno', extent=(df_umap_Pre['UMAP1'].min(), df_umap_Pre['UMAP1'].max(), df_umap_Pre['UMAP2'].min(), df_umap_Pre['UMAP2'].max()))
        hb_1hr = ax[1].hexbin(x=df_umap_1hr['UMAP1'], y=df_umap_1hr['UMAP2'], bins='log', cmap='inferno', extent=(df_umap_Pre['UMAP1'].min(), df_umap_Pre['UMAP1'].max(), df_umap_Pre['UMAP2'].min(), df_umap_Pre['UMAP2'].max()))
        hb_gen = ax[2].hexbin(x=df_umap_gen['UMAP1'], y=df_umap_gen['UMAP2'], bins='log', cmap='inferno', extent=(df_umap_Pre['UMAP1'].min(), df_umap_Pre['UMAP1'].max(), df_umap_Pre['UMAP2'].min(), df_umap_Pre['UMAP2'].max()))
        hb_gen_test = ax[3].hexbin(x=df_umap_gen_test['UMAP1'], y=df_umap_gen_test['UMAP2'], bins='log', cmap='inferno', extent=(df_umap_Pre['UMAP1'].min(), df_umap_Pre['UMAP1'].max(), df_umap_Pre['UMAP2'].min(), df_umap_Pre['UMAP2'].max()))
        max_ploy_count = max(max(hb_Pre.get_array()),max(hb_1hr.get_array()),max(hb_gen.get_array()))
        plt.close()

        fig, ax = plt.subplots(nrows=1, ncols=4, sharex=False, sharey=False, figsize=(22,4))
        # -6, 10, -6, 10
        hb_Pre = ax[0].hexbin(x=df_umap_Pre['UMAP1'], y=df_umap_Pre['UMAP2'], bins='log', cmap='inferno', vmax=max_ploy_count, extent=(-6, 14, -6, 14))
        hb_1hr = ax[1].hexbin(x=df_umap_1hr['UMAP1'], y=df_umap_1hr['UMAP2'], bins='log', cmap='inferno', vmax=max_ploy_count, extent=(-6, 14, -6, 14))
        hb_gen = ax[2].hexbin(x=df_umap_gen['UMAP1'], y=df_umap_gen['UMAP2'], bins='log', cmap='inferno', vmax=max_ploy_count, extent=(-6, 14, -6, 14))
        hb_gen_test = ax[3].hexbin(x=df_umap_gen_test['UMAP1'], y=df_umap_gen_test['UMAP2'], bins='log', cmap='inferno', vmax=max_ploy_count, extent=(-6, 14, -6, 14))
       
        ax[0].set_title('{} GT Pre'.format(patient_id))  
        ax[1].set_title('{} GT 1hr'.format(patient_id))  
        ax[2].set_title('{} train_GEN 1hr'.format(patient_id))
        ax[3].set_title('{} test_GEN 1hr'.format(patient_id))
       
        plt.savefig(os.path.join(current_file_path,'../raw_data_plots/plots/UMAP_raw_Pre_1hr/gen_train_and_test_all_models/Hexbin_plot_UMAP_Pre_1hr_gen_train_and_test_{}_all_models.pdf'.format(patient_id)), format='pdf', dpi=600)
        plt.close()

        #plot UMAP version 2
        df_umap_Pre = pd.DataFrame(data=my_umap.transform(df_all_gt_Pre_p), columns=['UMAP1', 'UMAP2'])
        df_umap_1hr = pd.DataFrame(data=my_umap.transform(df_all_gt_1hr_p), columns=['UMAP1', 'UMAP2'])
        df_umap_gen_test = pd.DataFrame(data=my_umap.transform(df_sample_gen_test), columns=['UMAP1', 'UMAP2'])
        
        ################ UMAP hexbin plot #########################
        fig, ax = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False, figsize=(17,4))
        hb_Pre = ax[0].hexbin(x=df_umap_Pre['UMAP1'], y=df_umap_Pre['UMAP2'], bins='log', cmap='inferno', extent=(df_umap_Pre['UMAP1'].min(), df_umap_Pre['UMAP1'].max(), df_umap_Pre['UMAP2'].min(), df_umap_Pre['UMAP2'].max()))
        hb_1hr = ax[1].hexbin(x=df_umap_1hr['UMAP1'], y=df_umap_1hr['UMAP2'], bins='log', cmap='inferno', extent=(df_umap_Pre['UMAP1'].min(), df_umap_Pre['UMAP1'].max(), df_umap_Pre['UMAP2'].min(), df_umap_Pre['UMAP2'].max()))
        hb_gen_test = ax[2].hexbin(x=df_umap_gen_test['UMAP1'], y=df_umap_gen_test['UMAP2'], bins='log', cmap='inferno', extent=(df_umap_Pre['UMAP1'].min(), df_umap_Pre['UMAP1'].max(), df_umap_Pre['UMAP2'].min(), df_umap_Pre['UMAP2'].max()))
        max_ploy_count = max(max(hb_Pre.get_array()),max(hb_1hr.get_array()),max(hb_gen_test.get_array()))
        plt.close()

        fig, ax = plt.subplots(nrows=1, ncols=4, sharex=False, sharey=False, figsize=(22,4))
        # -6, 10, -6, 10
        hb_Pre = ax[0].hexbin(x=df_umap_Pre['UMAP1'], y=df_umap_Pre['UMAP2'], bins='log', cmap='inferno', vmax=max_ploy_count, extent=(-6, 14, -6, 14))
        hb_1hr = ax[1].hexbin(x=df_umap_1hr['UMAP1'], y=df_umap_1hr['UMAP2'], bins='log', cmap='inferno', vmax=max_ploy_count, extent=(-6, 14, -6, 14))
        hb_gen_test = ax[2].hexbin(x=df_umap_gen_test['UMAP1'], y=df_umap_gen_test['UMAP2'], bins='log', cmap='inferno', vmax=max_ploy_count, extent=(-6, 14, -6, 14))
       
        ax[0].set_title('{} GT Pre'.format(patient_id))  
        ax[1].set_title('{} GT 1hr'.format(patient_id))  
        ax[2].set_title('{} test_GEN 1hr'.format(patient_id))
       
        plt.savefig(os.path.join(current_file_path,'../raw_data_plots/plots/UMAP_raw_Pre_1hr/gen_test_all_models/Hexbin_plot_UMAP_Pre_1hr_gen_train_and_test_{}_all_models.pdf'.format(patient_id)), format='pdf', dpi=600)
        plt.close()

        

def one_gen_all_models_train_test_donut_PCA_colored_by_kmeans(current_file_path): 
    # plot PCA colored by kmeans for generated samples from a patient using all models that had the patient as train 
    # also plot PCA colored by kmeans for generated samples from a patient using all models that had the patient as test 

    random.seed(42)
    all_patients = pickle.load(open(os.path.join(current_file_path,'../EnGen_train_iterations/training_data/all_patient_ids.pkl'), 'rb'))['all_patient_ids']
    column_labels = ['149Sm_CREB', '150Nd_STAT5', '151Eu_p38', '153Eu_STAT1', '154Sm_STAT3', '155Gd_S6', '159Tb_MAPKAPK2', '164Dy_IkB', '166Er_NFkB', '167Er_ERK', '168Er_pSTAT6', '113In_CD235ab_CD61', '115In_CD45', '143Nd_CD45RA', '139La_CD66',
                    '141Pr_CD7', '142Nd_CD19', '144Nd_CD11b', '145Nd_CD4', '146Nd_CD8a', '147Sm_CD11c', '148Nd_CD123', '156Gd_CD24', '157Gd_CD161', '158Gd_CD33', '165Ho_CD16', '169Tm_CD25', '170Er_CD3', '171Yb_CD27', '172Yb_CD15',
                    '173Yb_CCR2', '175Lu_CD14',	'176Lu_CD56', '160Gd_Tbet', '162Dy_FoxP3', '152Sm_TCRgd', '174Yb_HLADR']
    df_all_gt_Pre_1hr = pd.read_csv(os.path.join(current_file_path,'../data/gates/gt_Pre_1hr_all_patients.csv'))
    df_all_gt_Pre =  df_all_gt_Pre_1hr[df_all_gt_Pre_1hr['timepoint']=='Pre']
    df_all_gt_1hr =  df_all_gt_Pre_1hr[df_all_gt_Pre_1hr['timepoint']=='1hr']
    df_all_gt_Pre_1hr.drop(['timepoint', 'patient_id'], axis=1, inplace=True)
    df_all_gt_Pre_1hr = df_all_gt_Pre_1hr.sample(n=20000, random_state=42)
    df_all_gt_Pre_1hr.reset_index(drop=True, inplace=True)
    scaler = pickle.load(open(os.path.join(current_file_path,'../data/gates/All_patients_gt_Pre_1hr_scaler_pca.pkl'), 'rb'))['all_gt_Pre_1hr_scaler']
    pca = pickle.load(open(os.path.join(current_file_path,'../data/gates/All_patients_gt_Pre_1hr_scaler_pca.pkl'), 'rb'))['all_gt_Pre_1hr_pca']
    df_all_gt_Pre_1hr.loc[:,:] = scaler.transform(df_all_gt_Pre_1hr.values)
    df_pca_Pre_1hr = pd.DataFrame(data=pca.transform(df_all_gt_Pre_1hr), columns=['PC1', 'PC2'])

    kmeans = MiniBatchKMeans(n_clusters=3, random_state=42, batch_size=2000).fit(df_all_gt_Pre_1hr)
    pickle.dump({'kmeans':kmeans}, open(os.path.join(current_file_path,'../raw_data_plots/plots/raw_data_kmeans_3.pkl'), 'wb'))

    plot_patient = random.sample(all_patients, 1)[0] # select one patient randomly for plotting in Fig2 hexbin
    print('plot_patient = {}'.format(plot_patient))
    
    for patient_id in [plot_patient]: # only generate the plot_patient
    
        df_all_gt_Pre_p =  df_all_gt_Pre[df_all_gt_Pre['patient_id']==patient_id]
        df_all_gt_1hr_p =  df_all_gt_1hr[df_all_gt_1hr['patient_id']==patient_id]
        
        df_all_gt_Pre_p.drop(['timepoint', 'patient_id'], axis=1, inplace=True)
        df_all_gt_1hr_p.drop(['timepoint', 'patient_id'], axis=1, inplace=True)  
        
        df_all_gt_Pre_p = df_all_gt_Pre_p.sample(frac=1, random_state=42)
        df_all_gt_1hr_p = df_all_gt_1hr_p.sample(frac=1, random_state=42)
        
        df_all_gt_Pre_p.reset_index(drop=True, inplace=True)
        df_all_gt_1hr_p.reset_index(drop=True, inplace=True)

        df_sample_gen_train = pd.read_csv(os.path.join(current_file_path,'../data/gates/gen_train/gen_train_{}_all_models_inv_scaled_inv_arcsinh.csv'.format(patient_id)))
        df_sample_gen_test = pd.read_csv(os.path.join(current_file_path,'../data/gates/gen_test/gen_test_{}_all_models_inv_scaled_inv_arcsinh.csv'.format(patient_id)))
        assert len(column_labels)==len(df_sample_gen_train.columns.values), 'Columns do not match!'
        assert len(column_labels)==len(df_sample_gen_test.columns.values), 'Columns do not match!'
        df_sample_gen_train.columns = column_labels
        df_sample_gen_test.columns = column_labels
        df_sample_gen_train = df_sample_gen_train.sample(n=20000, random_state=42)
        df_sample_gen_test = df_sample_gen_test.sample(n=20000, random_state=42)
        df_sample_gen_train.reset_index(drop=True, inplace=True)
        df_sample_gen_test.reset_index(drop=True, inplace=True)

        df_all_gt_Pre_p.loc[:,:] = scaler.transform(df_all_gt_Pre_p.values)
        df_all_gt_1hr_p.loc[:,:] = scaler.transform(df_all_gt_1hr_p.values)
        df_sample_gen_train.loc[:,:] = scaler.transform(df_sample_gen_train.values)
        df_sample_gen_test.loc[:,:] = scaler.transform(df_sample_gen_test.values)
        
        #plot PCA
        df_pca_Pre = pd.DataFrame(data=pca.transform(df_all_gt_Pre_p), columns=['PC1', 'PC2'])
        df_pca_1hr = pd.DataFrame(data=pca.transform(df_all_gt_1hr_p), columns=['PC1', 'PC2'])
        df_pca_gen_train = pd.DataFrame(data=pca.transform(df_sample_gen_train), columns=['PC1', 'PC2'])
        df_pca_gen_test = pd.DataFrame(data=pca.transform(df_sample_gen_test), columns=['PC1', 'PC2'])

        # color nodes based on cluster ids
        colors = ['#81ecec', '#74b9ff', '#a29bfe']
        labels_Pre = list(kmeans.predict(df_all_gt_Pre_p))
        labels_1hr = list(kmeans.predict(df_all_gt_1hr_p))
        labels_gen_train = list(kmeans.predict(df_sample_gen_train))
        labels_gen_test = list(kmeans.predict(df_sample_gen_test))
        sizes_Pre=[labels_Pre.count(0), labels_Pre.count(1), labels_Pre.count(2)]
        sizes_1hr =[labels_1hr.count(0), labels_1hr.count(1), labels_1hr.count(2)]
        sizes_gen_train=[labels_gen_train.count(0), labels_gen_train.count(1), labels_gen_train.count(2)]
        sizes_gen_test=[labels_gen_test.count(0), labels_gen_test.count(1), labels_gen_test.count(2)]
        #plot donut 
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5), subplot_kw=dict(aspect="equal"))
        wedges, texts = ax.pie(sizes_Pre, wedgeprops=dict(width=0.5), startangle=-40, colors=colors)
        plt.savefig(os.path.join(current_file_path,'../raw_data_plots/plots/PCA_raw_Pre_1hr/gen_train_all_models/Donut_plot_kmeans_Pre_{}.pdf'.format(patient_id)), format='pdf', dpi=600)
        plt.close()
       
        #plot donut 
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5), subplot_kw=dict(aspect="equal"))
        wedges, texts = ax.pie(sizes_1hr, wedgeprops=dict(width=0.5), startangle=-40, colors=colors)
        plt.savefig(os.path.join(current_file_path,'../raw_data_plots/plots/PCA_raw_Pre_1hr/gen_train_all_models/Donut_plot_kmeans_1hr_{}.pdf'.format(patient_id)), format='pdf', dpi=600)
        plt.close()

        #plot donut 
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5), subplot_kw=dict(aspect="equal"))
        wedges, texts = ax.pie(sizes_gen_train, wedgeprops=dict(width=0.5), startangle=-40, colors=colors)
        plt.savefig(os.path.join(current_file_path,'../raw_data_plots/plots/PCA_raw_Pre_1hr/gen_train_all_models/Donut_plot_kmeans_gen_train_{}.pdf'.format(patient_id)), format='pdf', dpi=600)
        plt.close()

        #plot donut 
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5), subplot_kw=dict(aspect="equal"))
        wedges, texts = ax.pie(sizes_gen_test, wedgeprops=dict(width=0.5), startangle=-40, colors=colors)
        plt.savefig(os.path.join(current_file_path,'../raw_data_plots/plots/PCA_raw_Pre_1hr/gen_train_all_models/Donut_plot_kmeans_gen_test_{}.pdf'.format(patient_id)), format='pdf', dpi=600)
        plt.close()
        
        # colored PCA plot
        color_Pre = [colors[i] for i in labels_Pre]
        color_1hr = [colors[i] for i in labels_1hr]
        color_gen_train = [colors[i] for i in labels_gen_train]
        color_gen_test = [colors[i] for i in labels_gen_test]
        
        fig, ax = plt.subplots(nrows=1, ncols=4, sharex=False, sharey=False, figsize=(21,4))

        ################ PCA scatter plot #########################
        df_pca_Pre.plot(kind='scatter', x='PC1', y='PC2', ax=ax[0], c=color_Pre, alpha=1, s=1)
        df_pca_1hr.plot(kind='scatter', x='PC1', y='PC2', ax=ax[1], c=color_1hr, alpha=1, s=1)
        df_pca_gen_train.plot(kind='scatter', x='PC1', y='PC2', ax=ax[2], c=color_gen_train, s=1)
        df_pca_gen_test.plot(kind='scatter', x='PC1', y='PC2', ax=ax[3], c=color_gen_test, s=1)
        ax[0].set_title('{} GT Pre'.format(patient_id))  
        ax[1].set_title('{} GT 1hr'.format(patient_id))  
        ax[2].set_title('{} train_GEN 1hr'.format(patient_id))
        ax[3].set_title('{} test_GEN 1hr'.format(patient_id))
        ax[0].set_ylim((-10,12))
        ax[0].set_xlim((-10,12))
        ax[1].set_ylim((-10,12))
        ax[1].set_xlim((-10,12))
        ax[2].set_ylim((-10,12))
        ax[2].set_xlim((-10,12))
        ax[3].set_ylim((-10,12))
        ax[3].set_xlim((-10,12))
        # plt.show()
        plt.savefig(os.path.join(current_file_path,'../raw_data_plots/plots/PCA_raw_Pre_1hr/Scatter_plot_PCA_Pre_1hr_gen_train_{}_all_models_kmeans_colored.pdf'.format(patient_id)), format='pdf', dpi=600)

        
def one_gt_matched_sankey_colored_by_kmeans(current_file_path): 
    # cluster Pre and 1hr matched gt by kmeans from a patient look at matched cells and plot a sankey

    random.seed(42)
    all_patients = pickle.load(open(os.path.join(current_file_path,'../EnGen_train_iterations/training_data/all_patient_ids.pkl'), 'rb'))['all_patient_ids']
    column_labels = ['149Sm_CREB', '150Nd_STAT5', '151Eu_p38', '153Eu_STAT1', '154Sm_STAT3', '155Gd_S6', '159Tb_MAPKAPK2', '164Dy_IkB', '166Er_NFkB', '167Er_ERK', '168Er_pSTAT6', '113In_CD235ab_CD61', '115In_CD45', '143Nd_CD45RA', '139La_CD66',
                    '141Pr_CD7', '142Nd_CD19', '144Nd_CD11b', '145Nd_CD4', '146Nd_CD8a', '147Sm_CD11c', '148Nd_CD123', '156Gd_CD24', '157Gd_CD161', '158Gd_CD33', '165Ho_CD16', '169Tm_CD25', '170Er_CD3', '171Yb_CD27', '172Yb_CD15',
                    '173Yb_CCR2', '175Lu_CD14',	'176Lu_CD56', '160Gd_Tbet', '162Dy_FoxP3', '152Sm_TCRgd', '174Yb_HLADR']

    kmeans = pickle.load(open(os.path.join(current_file_path,'../raw_data_plots/plots/raw_data_kmeans_3.pkl'), 'rb'))['kmeans']
    scaler = pickle.load(open(os.path.join(current_file_path,'../data/gates/All_patients_gt_Pre_1hr_scaler_pca.pkl'), 'rb'))['all_gt_Pre_1hr_scaler']
    
    plot_patient = random.sample(all_patients, 1)[0] # select one patient randomly for plotting in Fig2 hexbin
    print('plot_patient = {}'.format(plot_patient))
    
    for patient_id in [plot_patient]: # only generate the plot_patient

        df_Pre_1hr_matched_p = pd.read_csv(os.path.join(current_file_path,'../EnGen_train_iterations/training_data/iter_00/Func_Pheno_45k_scaled_with_Pre_1hr_tps_source_Pre_target_1hr_matched.csv'))
        df_Pre_1hr_matched_p = df_Pre_1hr_matched_p[df_Pre_1hr_matched_p['patient_id']==patient_id]
        df_Pre_1hr_matched_p.drop('patient_id', axis=1, inplace=True)

        df_Pre_1hr_matched_p.reset_index(drop=True, inplace=True)
        assert len(column_labels)*2==len(df_Pre_1hr_matched_p.columns.values), 'Column sizes do not match!'
        df_Pre_matched_p = df_Pre_1hr_matched_p.iloc[:,:len(column_labels)]
        df_Pre_matched_p.columns = column_labels
        df_1hr_matched_p = df_Pre_1hr_matched_p.iloc[:,len(column_labels):]
        df_1hr_matched_p.columns = column_labels
        df_p = pd.concat([df_Pre_matched_p, df_1hr_matched_p], axis=0)

        df_Pre_1hr_matched_p['cluster_id_Pre'] = kmeans.predict(df_Pre_1hr_matched_p.iloc[:,:len(column_labels)])
        df_Pre_1hr_matched_p['cluster_id_1hr'] = kmeans.predict(df_Pre_1hr_matched_p.iloc[:,len(column_labels):-1]) # excluding the cluster_id_Pre column that was just added

        colors=['#81ecec', '#74b9ff', '#a29bfe']
        hex_color = [i.lstrip("#") for i in colors]
        colors = ['rgba({}, {}, {}, {})'.format(int(i[0:2], 16), int(i[2:4], 16), int(i[4:6], 16), 0.2) for i in hex_color]
        label = ['', '', '', '', '', '']
        node_colors = ['#81ecec', '#74b9ff', '#a29bfe', '#81ecec', '#74b9ff', '#a29bfe']
        source = []
        target = []
        value = []
        link_colors = []
        for id_Pre in range(3):
            for id_1hr in range(3,7):
                df_node = df_Pre_1hr_matched_p[df_Pre_1hr_matched_p['cluster_id_Pre']==id_Pre]
                source.append(id_Pre)
                target.append(id_1hr)
                value.append(df_node[df_node['cluster_id_1hr']==id_1hr-3].shape[0]/df_Pre_1hr_matched_p.shape[0]*100)
                link_colors.append(colors[id_Pre])
        
        link = dict(source = source, target = target, value = value, color=link_colors)
        node = dict(label = label, pad=50, thickness=15, color=node_colors)
        data = go.Sankey(link = link, node=node)
        # plot using plotly
        fig = go.Figure(data)
        fig.update_layout(autosize=False, width=500, height=500)
        fig.write_image(os.path.join(current_file_path,'../raw_data_plots/plots/PCA_raw_Pre_1hr/Sankey_plot_matching_{}.png'.format(patient_id)))


def inv_arcsinh_transformation(x):
    a = 0
    b = 1/5
    c = 0

    return (np.sinh(x-c)-a)/b


def gen_one_model_EnGenVAE_PCA(current_file_path, iter_id=0):
    
    #plot PCA for generated samples from an EnGenVAE model 
    random.seed(42)
    all_patients = pickle.load(open(os.path.join(current_file_path,'../EnGen_train_iterations/training_data/all_patient_ids.pkl'), 'rb'))['all_patient_ids']
    column_labels = ['149Sm_CREB', '150Nd_STAT5', '151Eu_p38', '153Eu_STAT1', '154Sm_STAT3', '155Gd_S6', '159Tb_MAPKAPK2', '164Dy_IkB', '166Er_NFkB', '167Er_ERK', '168Er_pSTAT6', '113In_CD235ab_CD61', '115In_CD45', '143Nd_CD45RA', '139La_CD66',
                    '141Pr_CD7', '142Nd_CD19', '144Nd_CD11b', '145Nd_CD4', '146Nd_CD8a', '147Sm_CD11c', '148Nd_CD123', '156Gd_CD24', '157Gd_CD161', '158Gd_CD33', '165Ho_CD16', '169Tm_CD25', '170Er_CD3', '171Yb_CD27', '172Yb_CD15',
                    '173Yb_CCR2', '175Lu_CD14',	'176Lu_CD56', '160Gd_Tbet', '162Dy_FoxP3', '152Sm_TCRgd', '174Yb_HLADR']
    df_all_gt_Pre_1hr = pd.read_csv(os.path.join(current_file_path,'../data/gates/gt_Pre_1hr_all_patients.csv'))
    df_all_gt_Pre =  df_all_gt_Pre_1hr[df_all_gt_Pre_1hr['timepoint']=='Pre']
    df_all_gt_1hr =  df_all_gt_Pre_1hr[df_all_gt_Pre_1hr['timepoint']=='1hr']
    df_all_gt_Pre_1hr.drop(['timepoint', 'patient_id'], axis=1, inplace=True)
    df_all_gt_Pre_1hr = df_all_gt_Pre_1hr.sample(n=20000, random_state=42)
    df_all_gt_Pre_1hr.reset_index(drop=True, inplace=True)
    scaler = pickle.load(open(os.path.join(current_file_path,'../data/gates/All_patients_gt_Pre_1hr_scaler_pca.pkl'), 'rb'))['all_gt_Pre_1hr_scaler']
    pca = pickle.load(open(os.path.join(current_file_path,'../data/gates/All_patients_gt_Pre_1hr_scaler_pca.pkl'), 'rb'))['all_gt_Pre_1hr_pca']
    df_all_gt_Pre_1hr.loc[:,:] = scaler.transform(df_all_gt_Pre_1hr.values)
    df_pca_Pre_1hr = pd.DataFrame(data=pca.transform(df_all_gt_Pre_1hr), columns=['PC1', 'PC2'])

    
    print('plot model iter = {}'.format(iter_id))
    df_all_gt_Pre_new = df_all_gt_Pre.sample(n=20000, random_state=42)
    df_all_gt_1hr_new = df_all_gt_1hr.sample(n=20000, random_state=42)
    
    df_all_gt_Pre_new.drop(['timepoint', 'patient_id'], axis=1, inplace=True)
    df_all_gt_1hr_new.drop(['timepoint', 'patient_id'], axis=1, inplace=True)  
    
    df_all_gt_Pre_new = df_all_gt_Pre_new.sample(frac=1, random_state=42)
    df_all_gt_1hr_new = df_all_gt_1hr_new.sample(frac=1, random_state=42)
    
    df_all_gt_Pre_new.reset_index(drop=True, inplace=True)
    df_all_gt_1hr_new.reset_index(drop=True, inplace=True)

    df_all_gt_Pre_new.loc[:,:] = scaler.transform(df_all_gt_Pre_new.values)
    df_all_gt_1hr_new.loc[:,:] = scaler.transform(df_all_gt_1hr_new.values)
    df_pca_Pre = pd.DataFrame(data=pca.transform(df_all_gt_Pre_new), columns=['PC1', 'PC2'])
    df_pca_1hr = pd.DataFrame(data=pca.transform(df_all_gt_1hr_new), columns=['PC1', 'PC2'])
        

    if iter_id < 10:
            iter_folder = 'iter_0{}'.format(iter_id)
    else:
        iter_folder = 'iter_{}'.format(iter_id)

    for i in range(5): # biospecimens generated using the iter_id model
        print(i)
        df_sample_gen = pd.read_csv(os.path.join(current_file_path,'../EnGenVAE_train_iterations/engen_vae_output/{0:}/generated/generated_{0:}_{1:}.csv'.format(iter_folder,i)))
        df_sample_gen = df_sample_gen.applymap(inv_arcsinh_transformation)
        df_sample_gen = df_sample_gen.sample(n=20000, random_state=42)
        df_sample_gen[df_sample_gen < 0] = 0
        assert len(column_labels)==len(df_sample_gen.columns.values), 'Columns do not match!'
        df_sample_gen.columns = column_labels
        df_sample_gen.reset_index(drop=True, inplace=True)

        
        df_sample_gen.loc[:,:] = scaler.transform(df_sample_gen.values)
        

        df_pca_gen = pd.DataFrame(data=pca.transform(df_sample_gen), columns=['PC1', 'PC2'])
        
        ################ PCA hexbin plot #########################
        fig, ax = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False, figsize=(17,4))
        hb_Pre = ax[0].hexbin(x=df_pca_Pre['PC1'], y=df_pca_Pre['PC2'], bins='log', cmap='inferno', extent=(df_pca_Pre['PC1'].min(), df_pca_Pre['PC1'].max(), df_pca_Pre['PC2'].min(), df_pca_Pre['PC2'].max()))
        hb_1hr = ax[1].hexbin(x=df_pca_1hr['PC1'], y=df_pca_1hr['PC2'], bins='log', cmap='inferno', extent=(df_pca_Pre['PC1'].min(), df_pca_Pre['PC1'].max(), df_pca_Pre['PC2'].min(), df_pca_Pre['PC2'].max()))
        hb_gen = ax[2].hexbin(x=df_pca_gen['PC1'], y=df_pca_gen['PC2'], bins='log', cmap='inferno', extent=(df_pca_Pre['PC1'].min(), df_pca_Pre['PC1'].max(), df_pca_Pre['PC2'].min(), df_pca_Pre['PC2'].max()))
        max_ploy_count = max(max(hb_Pre.get_array()),max(hb_1hr.get_array()),max(hb_gen.get_array()))
        plt.close()
        fig, ax = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False, figsize=(17,4))
        hb_Pre = ax[0].hexbin(x=df_pca_Pre['PC1'], y=df_pca_Pre['PC2'], bins='log', cmap='inferno', vmax=max_ploy_count, extent=(-6, 10, -6, 10))
        hb_1hr = ax[1].hexbin(x=df_pca_1hr['PC1'], y=df_pca_1hr['PC2'], bins='log', cmap='inferno', vmax=max_ploy_count, extent=(-6, 10, -6, 10))
        hb_gen = ax[2].hexbin(x=df_pca_gen['PC1'], y=df_pca_gen['PC2'], bins='log', cmap='inferno', vmax=max_ploy_count, extent=(-6, 10, -6, 10))
        
        ax[0].set_title('{0:}_{1:} GT Pre'.format(iter_folder,i))  
        ax[1].set_title('{0:}_{1:} GT 1hr'.format(iter_folder,i))  
        ax[2].set_title('{0:}_{1:} GEN 1hr'.format(iter_folder,i))
        
        os.makedirs(os.path.join(current_file_path,'../raw_data_plots/plots/EnGenVAE_PCA_raw_Pre_1hr/gen_model_{0:}/'.format(iter_folder)),exist_ok=True)
        plt.savefig(os.path.join(current_file_path,'../raw_data_plots/plots/EnGenVAE_PCA_raw_Pre_1hr/gen_model_{0:}/Hexbin_plot_PCA_Pre_1hr_gen_model_{0:}_{1:}.pdf'.format(iter_folder,i)), format='pdf', dpi=600)
        plt.close()



def gen_one_model_EnGenVAE_UMAP(current_file_path, iter_id=0):
    
    #plot UMAP for generated samples from one EnGenVAE model 
    random.seed(42)
    all_patients = pickle.load(open(os.path.join(current_file_path,'../EnGen_train_iterations/training_data/all_patient_ids.pkl'), 'rb'))['all_patient_ids']
    column_labels = ['149Sm_CREB', '150Nd_STAT5', '151Eu_p38', '153Eu_STAT1', '154Sm_STAT3', '155Gd_S6', '159Tb_MAPKAPK2', '164Dy_IkB', '166Er_NFkB', '167Er_ERK', '168Er_pSTAT6', '113In_CD235ab_CD61', '115In_CD45', '143Nd_CD45RA', '139La_CD66',
                    '141Pr_CD7', '142Nd_CD19', '144Nd_CD11b', '145Nd_CD4', '146Nd_CD8a', '147Sm_CD11c', '148Nd_CD123', '156Gd_CD24', '157Gd_CD161', '158Gd_CD33', '165Ho_CD16', '169Tm_CD25', '170Er_CD3', '171Yb_CD27', '172Yb_CD15',
                    '173Yb_CCR2', '175Lu_CD14',	'176Lu_CD56', '160Gd_Tbet', '162Dy_FoxP3', '152Sm_TCRgd', '174Yb_HLADR']
    df_all_gt_Pre_1hr = pd.read_csv(os.path.join(current_file_path,'../data/gates/gt_Pre_1hr_all_patients.csv'))
    df_all_gt_Pre =  df_all_gt_Pre_1hr[df_all_gt_Pre_1hr['timepoint']=='Pre']
    df_all_gt_1hr =  df_all_gt_Pre_1hr[df_all_gt_Pre_1hr['timepoint']=='1hr']
    df_all_gt_Pre_1hr.drop(['timepoint', 'patient_id'], axis=1, inplace=True)
    df_all_gt_Pre_1hr = df_all_gt_Pre_1hr.sample(n=20000, random_state=42)
    df_all_gt_Pre_1hr.reset_index(drop=True, inplace=True)
    scaler = pickle.load(open(os.path.join(current_file_path,'../data/gates/All_patients_gt_Pre_1hr_scaler_pca.pkl'), 'rb'))['all_gt_Pre_1hr_scaler']
    pca = pickle.load(open(os.path.join(current_file_path,'../data/gates/All_patients_gt_Pre_1hr_scaler_pca.pkl'), 'rb'))['all_gt_Pre_1hr_pca']
    df_all_gt_Pre_1hr.loc[:,:] = scaler.transform(df_all_gt_Pre_1hr.values)
    df_pca_Pre_1hr = pd.DataFrame(data=pca.transform(df_all_gt_Pre_1hr), columns=['PC1', 'PC2'])

    my_umap = UMAP(n_neighbors=5,
                      min_dist=0.3,
                      metric='correlation').fit(df_all_gt_Pre_1hr)
    # my_umap = UMAP().fit(df_all_gt_Pre_1hr)
    
    print('plot model iter = {}'.format(iter_id))
    
    df_all_gt_Pre_new = df_all_gt_Pre.sample(n=20000, random_state=42)
    df_all_gt_1hr_new = df_all_gt_1hr.sample(n=20000, random_state=42)
    
    df_all_gt_Pre_new.drop(['timepoint', 'patient_id'], axis=1, inplace=True)
    df_all_gt_1hr_new.drop(['timepoint', 'patient_id'], axis=1, inplace=True)  
    
    df_all_gt_Pre_new = df_all_gt_Pre_new.sample(frac=1, random_state=42)
    df_all_gt_1hr_new = df_all_gt_1hr_new.sample(frac=1, random_state=42)
    
    df_all_gt_Pre_new.reset_index(drop=True, inplace=True)
    df_all_gt_1hr_new.reset_index(drop=True, inplace=True)

    df_all_gt_Pre_new.loc[:,:] = scaler.transform(df_all_gt_Pre_new.values)
    df_all_gt_1hr_new.loc[:,:] = scaler.transform(df_all_gt_1hr_new.values)
        
    df_umap_Pre = pd.DataFrame(data=my_umap.transform(df_all_gt_Pre_new), columns=['PC1', 'PC2'])
    df_umap_1hr = pd.DataFrame(data=my_umap.transform(df_all_gt_1hr_new), columns=['PC1', 'PC2'])
        

    if iter_id < 10:
            iter_folder = 'iter_0{}'.format(iter_id)
    else:
        iter_folder = 'iter_{}'.format(iter_id)
    for i in range(3): # biospecimens generated using the iter_id model
        print(i)
        df_sample_gen = pd.read_csv(os.path.join(current_file_path,'../EnGenVAE_train_iterations/engen_vae_output/{0:}/generated/generated_{0:}_{1:}.csv'.format(iter_folder,i)))
        df_sample_gen = df_sample_gen.applymap(inv_arcsinh_transformation)
        df_sample_gen = df_sample_gen.sample(n=20000, random_state=42)
        df_sample_gen[df_sample_gen < 0] = 0
        assert len(column_labels)==len(df_sample_gen.columns.values), 'Columns do not match!'
        df_sample_gen.columns = column_labels
        df_sample_gen.reset_index(drop=True, inplace=True)

        df_sample_gen.loc[:,:] = scaler.transform(df_sample_gen.values)
        
        # plot UMAP
        df_umap_gen = pd.DataFrame(data=my_umap.transform(df_sample_gen), columns=['PC1', 'PC2'])
        
        ################ UMAP hexbin plot #########################
        fig, ax = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False, figsize=(17,4))
        hb_Pre = ax[0].hexbin(x=df_umap_Pre['PC1'], y=df_umap_Pre['PC2'], bins='log', cmap='inferno', extent=(df_umap_Pre['PC1'].min(), df_umap_Pre['PC1'].max(), df_umap_Pre['PC2'].min(), df_umap_Pre['PC2'].max()))
        hb_1hr = ax[1].hexbin(x=df_umap_1hr['PC1'], y=df_umap_1hr['PC2'], bins='log', cmap='inferno', extent=(df_umap_Pre['PC1'].min(), df_umap_Pre['PC1'].max(), df_umap_Pre['PC2'].min(), df_umap_Pre['PC2'].max()))
        hb_gen = ax[2].hexbin(x=df_umap_gen['PC1'], y=df_umap_gen['PC2'], bins='log', cmap='inferno', extent=(df_umap_Pre['PC1'].min(), df_umap_Pre['PC1'].max(), df_umap_Pre['PC2'].min(), df_umap_Pre['PC2'].max()))
        max_ploy_count = max(max(hb_Pre.get_array()),max(hb_1hr.get_array()),max(hb_gen.get_array()))
        plt.close()
        fig, ax = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False, figsize=(17,4))
        hb_Pre = ax[0].hexbin(x=df_umap_Pre['PC1'], y=df_umap_Pre['PC2'], bins='log', cmap='inferno', vmax=max_ploy_count, extent=(-6, 15, -6, 14))
        hb_1hr = ax[1].hexbin(x=df_umap_1hr['PC1'], y=df_umap_1hr['PC2'], bins='log', cmap='inferno', vmax=max_ploy_count, extent=(-6, 15, -6, 14))
        hb_gen = ax[2].hexbin(x=df_umap_gen['PC1'], y=df_umap_gen['PC2'], bins='log', cmap='inferno', vmax=max_ploy_count, extent=(-6, 15, -6, 14))
        
        ax[0].set_title('{0:}_{1:} GT Pre'.format(iter_folder,i))  
        ax[1].set_title('{0:}_{1:} GT 1hr'.format(iter_folder,i))  
        ax[2].set_title('{0:}_{1:} GEN 1hr'.format(iter_folder,i))
        
        os.makedirs(os.path.join(current_file_path,'../raw_data_plots/plots/EnGenVAE_UMAP_raw_Pre_1hr/gen_model_{0:}/'.format(iter_folder)),exist_ok=True)
        plt.savefig(os.path.join(current_file_path,'../raw_data_plots/plots/EnGenVAE_UMAP_raw_Pre_1hr/gen_model_{0:}/Hexbin_plot_UMAP_Pre_1hr_gen_model_{0:}_{1:}.pdf'.format(iter_folder,i)), format='pdf', dpi=600)
        plt.close()


if __name__ == "__main__":
    
    current_file_path = os.path.abspath(os.path.dirname(__file__))
    # one_gen_all_models_train_PCA(current_file_path)
    # one_gen_all_models_test_PCA(current_file_path)
    # one_gen_all_models_train_and_test_UMAP(current_file_path)
    # one_gen_all_models_train_test_donut_PCA_colored_by_kmeans(current_file_path)
    # one_gt_matched_sankey_colored_by_kmeans(current_file_path)

    for iter_id in range(30):
        gen_one_model_EnGenVAE_PCA(current_file_path, iter_id=iter_id)
        gen_one_model_EnGenVAE_UMAP(current_file_path, iter_id=iter_id)





    
    
