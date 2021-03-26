import sys
import os
import pandas as pd 
import numpy as np 
import random
import re 
from matplotlib import pyplot as plt
from matplotlib import cm, ticker
import matplotlib.lines as mlines
import itertools
import pickle
import copy
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)


def arcsinh_transformation(x):
    a = 0
    b = 1/5
    c = 0
    return np.arcsinh(a+b*x)+c




def extract_feature(current_file_path, n_clusters=25):
    
    all_patient_ids = pickle.load(open(os.path.join(current_file_path,'../EnGen_train_iterations/training_data/all_patient_ids.pkl'), 'rb'))['all_patient_ids']
    
    column_labels = ['149Sm_CREB', '150Nd_STAT5', '151Eu_p38', '153Eu_STAT1', '154Sm_STAT3', '155Gd_S6', '159Tb_MAPKAPK2', '164Dy_IkB', '166Er_NFkB', '167Er_ERK', '168Er_pSTAT6', '113In_CD235ab_CD61', '115In_CD45', '143Nd_CD45RA', '139La_CD66',
                    '141Pr_CD7', '142Nd_CD19', '144Nd_CD11b', '145Nd_CD4', '146Nd_CD8a', '147Sm_CD11c', '148Nd_CD123', '156Gd_CD24', '157Gd_CD161', '158Gd_CD33', '165Ho_CD16', '169Tm_CD25', '170Er_CD3', '171Yb_CD27', '172Yb_CD15',
                    '173Yb_CCR2', '175Lu_CD14',	'176Lu_CD56', '160Gd_Tbet', '162Dy_FoxP3', '152Sm_TCRgd', '174Yb_HLADR']

    
    
    #learn scaler and kmeans on gt Pre and gt 1hr of all patients
    df_gt_st = pd.DataFrame() # both source and target
    for source_patient_id in all_patient_ids:
       
        df_sample_Pre = pd.read_csv(os.path.join(current_file_path,'../data/preprocessed/Pre/Func_Pheno_20k_{0:}_A_Pre.csv'.format(source_patient_id)))
        assert len(column_labels)==len(df_sample_Pre.columns.values), 'Columns do not match!'
        df_sample_Pre.columns = column_labels
        df_sample_Pre = df_sample_Pre.applymap(arcsinh_transformation)

        df_sample_1hr = pd.read_csv(os.path.join(current_file_path,'../data/preprocessed/1hr/Func_Pheno_20k_{0:}_A_1hr.csv'.format(source_patient_id)))
        assert len(column_labels)==len(df_sample_1hr.columns.values), 'Columns do not match!'
        df_sample_1hr.columns = column_labels
        df_sample_1hr = df_sample_1hr.applymap(arcsinh_transformation)

        df_gt_st = pd.concat([df_gt_st, df_sample_Pre, df_sample_1hr], axis=0) # concat source and target of all patients in one file

    f_gt_st = df_gt_st.sample(frac=1, random_state=42)
    df_gt_st.reset_index(inplace=True, drop=True)
    scaler = StandardScaler()
    #print('fitting scaler')
    scaler.fit(df_gt_st.iloc[:,:].values)
    df_gt_st.iloc[:,:] = scaler.transform(df_gt_st.iloc[:,:].values)
    #print('fitting kmeans')
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=10000).fit(df_gt_st.iloc[:,:])
    
    all_frequencies = []
    for source_patient_id in all_patient_ids:
        
        df_sample_Pre = pd.read_csv(os.path.join(current_file_path,'../data/preprocessed/Pre/Func_Pheno_20k_{0:}_A_Pre.csv'.format(source_patient_id)))
        assert len(column_labels)==len(df_sample_Pre.columns.values), 'Columns do not match!'
        df_sample_Pre.columns = column_labels
        df_sample_Pre = df_sample_Pre.applymap(arcsinh_transformation)
        df_sample_Pre.iloc[:,:] = scaler.transform(df_sample_Pre.values)   
        df_sample_Pre['cluster_label'] = kmeans.predict(df_sample_Pre)

        df_sample_1hr = pd.read_csv(os.path.join(current_file_path,'../data/preprocessed/1hr/Func_Pheno_20k_{0:}_A_1hr.csv'.format(source_patient_id)))
        assert len(column_labels)==len(df_sample_1hr.columns.values), 'Columns do not match!'
        df_sample_1hr.columns = column_labels
        df_sample_1hr = df_sample_1hr.applymap(arcsinh_transformation)
        df_sample_1hr.iloc[:,:] = scaler.transform(df_sample_1hr.values)         
        df_sample_1hr['cluster_label'] = kmeans.predict(df_sample_1hr)

        frequencies_row_Pre = []
        frequencies_row_Pre.append(source_patient_id) # patient_id
        frequencies_row_Pre.append('Pre') # timepoint
        frequencies_row_Pre.append('ground_truth') # data_type

        frequencies_row_1hr = []
        frequencies_row_1hr.append(source_patient_id) # patient_id
        frequencies_row_1hr.append('1hr') # timepoint
        frequencies_row_1hr.append('ground_truth') # data_type
  
        for i in range(n_clusters):
            df_file_cluster_label = df_sample_Pre[df_sample_Pre['cluster_label'] == i] 
            frequencies_row_Pre.append(df_file_cluster_label.shape[0])
            df_file_cluster_label = df_sample_1hr[df_sample_1hr['cluster_label'] == i] 
            frequencies_row_1hr.append(df_file_cluster_label.shape[0])
            
        all_frequencies.append(frequencies_row_Pre)
        all_frequencies.append(frequencies_row_1hr)

    n_iterations = 30

    for iter_id in range(n_iterations):
        #gen 1hr from Pre using the model from that iter

        all_frequencies_iter = copy.deepcopy(all_frequencies)

        if iter_id < 10:
            iter_folder = 'iter_0{}'.format(iter_id)
        else:
            iter_folder = 'iter_{}'.format(iter_id)
        
        p = pickle.load(open(os.path.join(current_file_path,'../EnGen_train_iterations/training_data/{}/scaler_kmeans_25.pkl'.format(iter_folder)), 'rb'))
        AE_train_ids = p['AE_train_ids']

        for source_patient_id in all_patient_ids:

            if source_patient_id in AE_train_ids:
                gen_path = os.path.join(current_file_path,'../EnGen_train_iterations/engen_output/{}/generated_train/'.format(iter_folder))
                data_type = 'gen_train'
            else:
                gen_path = os.path.join(current_file_path,'../EnGen_train_iterations/engen_output/{}/generated/'.format(iter_folder))
                data_type = 'gen_test'
            df_sample_gen = pd.read_csv(gen_path+'generated_{0:}.csv'.format(source_patient_id))
            df_sample_gen.reset_index(drop=True, inplace=True)
            df_sample_gen[df_sample_gen < 0] = 0     
            df_sample_gen.iloc[:,:] = scaler.transform(df_sample_gen.values)       
            assert len(column_labels)==len(df_sample_gen.columns.values), 'Columns do not match!'
            df_sample_gen.columns = column_labels
            
            df_sample_gen['cluster_label'] = kmeans.predict(df_sample_gen)

            frequencies_row_gen = []

            frequencies_row_gen.append(source_patient_id) # patient_id
            frequencies_row_gen.append('1hr') # timepoint
            frequencies_row_gen.append(data_type) # data_type

            
            # print(frequencies_row)
            for i in range(n_clusters):
                df_file_cluster_label = df_sample_gen[df_sample_gen['cluster_label'] == i] 
                frequencies_row_gen.append(df_file_cluster_label.shape[0])
                
            all_frequencies_iter.append(frequencies_row_gen)
                
        df_all_features = pd.DataFrame(data=all_frequencies_iter, columns=['patient_id','timepoint','data_type']+['c_{}'.format(i) for i in range(n_clusters)])

        df_all_features.to_csv(os.path.join(current_file_path,'../EnGen_train_iterations/engen_output/{0:}/Features_gt_Pre_gt_1hr_gen_1hr_test_train_kmeans{1:}_{0:}.csv'.format(iter_folder, n_clusters)), header=True, index=False)
   



def PCA_source_to_target_one_model(current_file_path, n_clusters=25):

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(6,6))
    n_iterations = 30

    
    for iter_id in range(n_iterations):
        if iter_id < 10:
            iter_id = 'iter_0{}'.format(iter_id)
        else:
            iter_id = 'iter_{}'.format(iter_id)
        p = pickle.load(open(os.path.join(current_file_path,'../EnGen_train_iterations/training_data/{}/scaler_kmeans_25.pkl'.format(iter_id)), 'rb'))
        df = pd.read_csv(os.path.join(current_file_path,'../EnGen_train_iterations/engen_output/{0:}/Features_gt_Pre_gt_1hr_gen_1hr_test_train_kmeans{1:}_{0:}.csv'.format(iter_id, n_clusters)))
        
        df_gt = df[df['data_type']=='ground_truth']
        df_gt.reset_index(inplace=True, drop=True)
        scaler = StandardScaler()
        #print('fitting scaler')
        scaler.fit(df_gt.iloc[:,3:].values)
        df_gt.iloc[:,3:] = scaler.transform(df_gt.iloc[:,3:].values)
        df.iloc[:,3:] = scaler.transform(df.iloc[:,3:].values)
        if iter_id == 'iter_00':
            pca = PCA(n_components=2)
            pca.fit(df_gt.iloc[:,3:])
            p['pca_Pre_1hr_gt'] = pca
            pickle.dump(p, open(os.path.join(current_file_path,'../EnGen_train_iterations/engen_output/pca_gt_{}_clusters.pkl'.format(n_clusters)), 'wb'))
        else:
            pca = pickle.load(open(os.path.join(current_file_path,'../EnGen_train_iterations/engen_output/pca_gt_{}_clusters.pkl'.format(n_clusters)), 'rb'))['pca_Pre_1hr_gt']

        ev0, ev1 = pca.explained_variance_ratio_
        df_pca = pd.DataFrame()
        df_pca['patient_id'] = df['patient_id']
        df_pca['timepoint'] = df['timepoint']
        df_pca['data_type'] = df['data_type']
        df_pca['PC1'] = [np.nan for _ in range(df_pca.shape[0])]
        df_pca['PC2'] = [np.nan for _ in range(df_pca.shape[0])]
        df_pca.iloc[:,3:] = pca.transform(df.iloc[:,3:])
  
        patient_ids = sorted(list(set(df_pca['patient_id'])))
        for idx, row in df_pca.iterrows():
            # print(row)
            style = {}
            style['s'] = 20
            if row['data_type'] == 'ground_truth' and iter_id == 'iter_00':
                
                style['facecolors']= 'none'
                style['marker']= 's'

                if row['timepoint'] == 'Pre':
                    style['edgecolors']= '#6FC2B3'
                else:
                    style['edgecolors']= '#2E7BA2'
            
                ax.scatter(row['PC1'], row['PC2'],**style)
            elif row['data_type'] == 'gen_train':
                style['marker']= 'o'
                style['s'] = 8
                style['facecolors']= 'none'
                if row['timepoint'] == 'Pre':
            
                    return
                else:
                    style['c']= '#e17055'
                ax.scatter(row['PC1'], row['PC2'],**style)
            elif row['data_type'] == 'gen_test':
                style['marker']= 'x'
                if row['timepoint'] == 'Pre':
                    # style['c']= 'black' # not applicable
                    return
                else:
                    style['c']= '#e17055'
                ax.scatter(row['PC1'], row['PC2'],**style)
        
        for patient_id in patient_ids:
            df_pca_p_gt = df_pca[(df_pca['patient_id']==patient_id) & (df_pca['data_type']=='ground_truth')]
            df_pca_p_gen = df_pca[(df_pca['patient_id']==patient_id) & (df_pca['data_type']!='ground_truth')]
            
        
            if df_pca_p_gt['timepoint'].values[0] == 'Pre':
                c = 'black'
            else:
                c = 'red'
                continue
            ax.arrow(df_pca_p_gt['PC1'].values[0],df_pca_p_gt['PC2'].values[0],
                (df_pca_p_gen['PC1'].values[0]-df_pca_p_gt['PC1'].values[0]),
                (df_pca_p_gen['PC2'].values[0]-df_pca_p_gt['PC2'].values[0]),
                width=0.02,color=c,head_length=0.0,head_width=0.0, alpha=0.03)

    ax.set_title('30 Iterations', fontsize=7)
    ax.set_xlabel('PC1 ({:.2f})'.format(ev0))
    ax.set_ylabel('PC2 ({:.2f})'.format(ev1))
    plt.setp(ax.get_xticklabels(), fontsize=5)
    plt.setp(ax.get_yticklabels(), fontsize=5)
    plt.xlim(-6, 7)
    plt.ylim(-4, 8)
    legend_elements = [mlines.Line2D([], [], color='#6FC2B3', marker='s', linestyle='None', markerfacecolor='None', markersize=5, label='GT Pre'),
                        mlines.Line2D([], [], color='#2E7BA2', marker='s', linestyle='None', markerfacecolor='None', markersize=5, label='GT 1hr'),
                        mlines.Line2D([], [], color='#e17055', marker='P', linestyle='None', markerfacecolor='#e17055', markersize=3, label='Gen train'),
                        mlines.Line2D([], [], color='#e17055', marker='x', linestyle='None', markerfacecolor='#e17055', markersize=5, label='Gen test')]

    ax.legend(handles=legend_elements, loc='best')
    # plt.show()
    
    plt.savefig(os.path.join(current_file_path,'../EnGen_train_iterations/engen_output/PCA_source_target.pdf'), format='pdf', dpi=600)

  

        



if __name__ == "__main__":
    
    current_file_path = os.path.abspath(os.path.dirname(__file__))

    # perform clustering and extract features for sampels 
    extract_feature(current_file_path)

    PCA_source_to_target_one_model(current_file_path)

  







    
    
