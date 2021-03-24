import sys
import os
import pandas as pd 
import numpy as np 
import random
import re 
from matplotlib import pyplot as plt
from matplotlib import cm
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



def build_pca_scaler_all_gt_Pre_1hr(current_file_path):

    # PCA and scaler will be used for PCA plots of raw data colored by gated cell types
    random.seed(42)
    df_all_gt_Pre_1hr = pd.read_csv(os.path.join(current_file_path,'../data/gates/gt_Pre_1hr_all_patients.csv'))
    df_all_gt_Pre =  df_all_gt_Pre_1hr[df_all_gt_Pre_1hr['timepoint']=='Pre']
    df_all_gt_1hr =  df_all_gt_Pre_1hr[df_all_gt_Pre_1hr['timepoint']=='1hr']
    df_all_gt_Pre_1hr.drop(['timepoint', 'patient_id'], axis=1, inplace=True)
    df_all_gt_Pre_1hr = df_all_gt_Pre_1hr.sample(n=20000, random_state=42)
    df_all_gt_Pre_1hr.reset_index(drop=True, inplace=True)
    scaler = StandardScaler()
    scaler.fit(df_all_gt_Pre_1hr.values) # fit on Pre and 1hr
    df_all_gt_Pre_1hr.loc[:,:] = scaler.transform(df_all_gt_Pre_1hr.values)
    pca = PCA(n_components=2, whiten=False)
    pca.fit(df_all_gt_Pre_1hr) # fit on Pre and 1hr
    p = {}
    p['all_gt_Pre_1hr_pca'] = pca
    p['all_gt_Pre_1hr_scaler'] = scaler
    pickle.dump(p, open(os.path.join(current_file_path,'../data/gates/All_patients_gt_Pre_1hr_scaler_pca.pkl'), 'wb'))


def Donut_PCA_colored_by_celltype(current_file_path): 
    # plot PCA colored by cell type for generated samples from a patient using all models that had the patient as train 
    # plot donut plots of cell type freqs
    # plot PCA colored by cell type for generated samples from a patient using all models that had the patient as test 
    # plot donut plots of cell type freqs

    random.seed(42)
    all_patients = pickle.load(open(os.path.join(current_file_path,'../EnGen_train_iterations/training_data/all_patient_ids.pkl'), 'rb'))['all_patient_ids']
    column_labels = ['149Sm_CREB', '150Nd_STAT5', '151Eu_p38', '153Eu_STAT1', '154Sm_STAT3', '155Gd_S6', '159Tb_MAPKAPK2', '164Dy_IkB', '166Er_NFkB', '167Er_ERK', '168Er_pSTAT6', '113In_CD235ab_CD61', '115In_CD45', '143Nd_CD45RA', '139La_CD66',
                    '141Pr_CD7', '142Nd_CD19', '144Nd_CD11b', '145Nd_CD4', '146Nd_CD8a', '147Sm_CD11c', '148Nd_CD123', '156Gd_CD24', '157Gd_CD161', '158Gd_CD33', '165Ho_CD16', '169Tm_CD25', '170Er_CD3', '171Yb_CD27', '172Yb_CD15',
                    '173Yb_CCR2', '175Lu_CD14',	'176Lu_CD56', '160Gd_Tbet', '162Dy_FoxP3', '152Sm_TCRgd', '174Yb_HLADR']

    scaler = pickle.load(open(os.path.join(current_file_path,'../data/gates/All_patients_gt_Pre_1hr_scaler_pca.pkl'), 'rb'))['all_gt_Pre_1hr_scaler']
    pca = pickle.load(open(os.path.join(current_file_path,'../data/gates/All_patients_gt_Pre_1hr_scaler_pca.pkl'), 'rb'))['all_gt_Pre_1hr_pca']

    plot_patient = random.sample(all_patients, 1)[0] # select one patient randomly for plotting
    print('plot_patient = {}'.format(plot_patient))

    
    celltypes = ['Bcells__Mem', 'Bcells__Naive', 'TCRgd', 'CD4pos__CD4posNaive', 'CD4pos__CD4posMem', 'CD8pos__CD8posNaive', 'CD8pos__CD8posMem', 'NKcells__MCcells', 'NKcells__NKcellsCD7pos'] # 9 cell types

    all_sizes_cols = ['patient_id', 'sample']
    all_sizes_cols.extend(celltypes)
    df_all_sizes = pd.DataFrame(columns = all_sizes_cols)
    df_all_sizes_percent = pd.DataFrame(columns = all_sizes_cols)
    df_all_sizes_diff = pd.DataFrame(columns = all_sizes_cols)
    
    # for patient_id in [plot_patient]: # only generate the plot_patient
    for patient_id in all_patients:  # generate the plots for all patients
        print('Patient = {}'.format(patient_id))
        df_gt_Pre_p =  pd.read_csv(os.path.join(current_file_path,'../data/gates/gt_Pre_all/patient_{}/cells_type_mononuclear_cells.csv'.format(patient_id)))
        df_gt_1hr_p =  pd.read_csv(os.path.join(current_file_path,'../data/gates/gt_1hr_all/patient_{}/cells_type_mononuclear_cells.csv'.format(patient_id)))

        df_gen_train_p = pd.read_csv(os.path.join(current_file_path,'../data/gates/gen_train_all_models/patient_{}/cells_type_mononuclear_cells.csv'.format(patient_id)))
        df_gen_test_p = pd.read_csv(os.path.join(current_file_path,'../data/gates/gen_test_all_models/patient_{}/cells_type_mononuclear_cells.csv'.format(patient_id)))
        assert len(column_labels)==len(df_gen_train_p.columns.values), 'Columns do not match!'
        df_gen_train_p.columns = column_labels
        assert len(column_labels)==len(df_gen_test_p.columns.values), 'Columns do not match!'
        df_gen_test_p.columns = column_labels

        # so far we got the gen and gt samples for a patient

        df_gt_Pre_p.loc[:,:] = scaler.transform(df_gt_Pre_p.values)
        df_gt_1hr_p.loc[:,:] = scaler.transform(df_gt_1hr_p.values)
        df_gen_train_p.loc[:,:] = scaler.transform(df_gen_train_p.values)
        df_gen_test_p.loc[:,:] = scaler.transform(df_gen_test_p.values)
        
        #plot PCA
        df_pca_Pre = pd.DataFrame(data=pca.transform(df_gt_Pre_p), columns=['PC1', 'PC2'])
        df_pca_1hr = pd.DataFrame(data=pca.transform(df_gt_1hr_p), columns=['PC1', 'PC2'])
        df_pca_gen_train = pd.DataFrame(data=pca.transform(df_gen_train_p), columns=['PC1', 'PC2'])
        df_pca_gen_test = pd.DataFrame(data=pca.transform(df_gen_test_p), columns=['PC1', 'PC2'])

        # plot colored PCAs
        colors = ['#ffeaa7', '#fab1a0', '#ff7675', '#fd79a8', '#55efc4', '#81ecec', '#74b9ff', '#a29bfe', '#b2bec3']
        
        gt_Pre_sizes = []
        gt_1hr_sizes = []
        gen_train_sizes = []
        gen_test_sizes = []
        
        fig, ax = plt.subplots(nrows=1, ncols=4, sharex=False, sharey=False, figsize=(18,3))
        for i, cell_type in enumerate(celltypes): 

            df_gt_Pre_p =  pd.read_csv(os.path.join(current_file_path,'../data/gates/gt_Pre_all/patient_{}/cells_type_{}.csv'.format(patient_id, cell_type)))
            df_gt_1hr_p =  pd.read_csv(os.path.join(current_file_path,'../data/gates/gt_1hr_all/patient_{}/cells_type_{}.csv'.format(patient_id, cell_type)))

            df_gen_train_p = pd.read_csv(os.path.join(current_file_path,'../data/gates/gen_train_all_models/patient_{}/cells_type_{}.csv'.format(patient_id, cell_type)))
            df_gen_test_p = pd.read_csv(os.path.join(current_file_path,'../data/gates/gen_test_all_models/patient_{}/cells_type_{}.csv'.format(patient_id, cell_type)))

            assert len(column_labels)==len(df_gen_train_p.columns.values), 'Columns do not match!'
            df_gen_train_p.columns = column_labels
            assert len(column_labels)==len(df_gen_test_p.columns.values), 'Columns do not match!'
            df_gen_test_p.columns = column_labels


            # so far we got the gen and gt samples for a patient

            df_gt_Pre_p.loc[:,:] = scaler.transform(df_gt_Pre_p.values)
            df_gt_1hr_p.loc[:,:] = scaler.transform(df_gt_1hr_p.values)
            df_gen_train_p.loc[:,:] = scaler.transform(df_gen_train_p.values)
            df_gen_test_p.loc[:,:] = scaler.transform(df_gen_test_p.values)
            
            #plot PCA
            df_pca_Pre = pd.DataFrame(data=pca.transform(df_gt_Pre_p), columns=['PC1', 'PC2'])
            df_pca_1hr = pd.DataFrame(data=pca.transform(df_gt_1hr_p), columns=['PC1', 'PC2'])
            df_pca_gen_train = pd.DataFrame(data=pca.transform(df_gen_train_p), columns=['PC1', 'PC2'])
            df_pca_gen_test = pd.DataFrame(data=pca.transform(df_gen_test_p), columns=['PC1', 'PC2'])

            gt_Pre_sizes.append(df_gt_Pre_p.shape[0])
            gt_1hr_sizes.append(df_gt_1hr_p.shape[0])
            gen_train_sizes.append(df_gen_train_p.shape[0])
            gen_test_sizes.append(df_gen_test_p.shape[0])

            ################ PCA scatter plot #########################
            df_pca_Pre.plot(kind='scatter', x='PC1', y='PC2', ax=ax[0], c=colors[i], alpha=0.1, s=1)
            df_pca_1hr.plot(kind='scatter', x='PC1', y='PC2', ax=ax[1], c=colors[i], alpha=0.1, s=1)
            df_pca_gen_train.plot(kind='scatter', x='PC1', y='PC2', ax=ax[2], c=colors[i], alpha=0.1, s=1)
            df_pca_gen_test.plot(kind='scatter', x='PC1', y='PC2', ax=ax[3], c=colors[i], alpha=0.1, s=1)

        row_all_sizes = [patient_id, 'Pre']+gt_Pre_sizes
        df_all_sizes.loc[df_all_sizes.shape[0],:] = row_all_sizes
        row_all_sizes = [patient_id, '1hr']+gt_1hr_sizes
        df_all_sizes.loc[df_all_sizes.shape[0],:] = row_all_sizes
        row_all_sizes = [patient_id, 'gen_test']+gen_test_sizes
        df_all_sizes.loc[df_all_sizes.shape[0],:] = row_all_sizes

        gt_Pre_sizes = [round(i/sum(gt_Pre_sizes)*100,2) for i in gt_Pre_sizes]
        gt_1hr_sizes = [round(i/sum(gt_1hr_sizes)*100,2) for i in gt_1hr_sizes]
        gen_test_sizes = [round(i/sum(gen_test_sizes)*100,2) for i in gen_test_sizes]
        
        row_all_sizes = [patient_id, 'Pre']+gt_Pre_sizes
        df_all_sizes_percent.loc[df_all_sizes_percent.shape[0],:] = row_all_sizes
        row_all_sizes = [patient_id, '1hr']+gt_1hr_sizes
        df_all_sizes_percent.loc[df_all_sizes_percent.shape[0],:] = row_all_sizes
        row_all_sizes = [patient_id, 'gen_test']+gen_test_sizes
        df_all_sizes_percent.loc[df_all_sizes_percent.shape[0],:] = row_all_sizes

        gt_1hr_diff = []
        gen_test_diff = []
        for i, j, k in zip(gt_Pre_sizes,gt_1hr_sizes,gen_test_sizes):
            gt_1hr_diff.append(round(j-i,1))
            gen_test_diff.append(round(k-i,1))
        
        row_all_sizes = [patient_id, '1hr']+gt_1hr_diff
        df_all_sizes_diff.loc[df_all_sizes_diff.shape[0],:] = row_all_sizes
        row_all_sizes = [patient_id, 'gen_test']+gen_test_diff
        df_all_sizes_diff.loc[df_all_sizes_diff.shape[0],:] = row_all_sizes

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

        os.makedirs(os.path.join(current_file_path,'../Rulebased_manual_gating/donut_plots/{0:}/'.format(patient_id)), exist_ok = True)
        plt.savefig(os.path.join(current_file_path,'../Rulebased_manual_gating/donut_plots/{0:}/Scatter_plot_PCA_Pre_1hr_gen_{0:}_all_models.pdf'.format(patient_id)), format='pdf', dpi=600)
        plt.close()
        #plot donut 
        fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20,5), subplot_kw=dict(aspect="equal"))

        wedges, texts = ax[0].pie(gt_Pre_sizes, wedgeprops=dict(width=0.5), startangle=-40, colors=colors)
        wedges, texts = ax[1].pie(gt_1hr_sizes, wedgeprops=dict(width=0.5), startangle=-40, colors=colors)
        wedges, texts = ax[2].pie(gen_train_sizes, wedgeprops=dict(width=0.5), startangle=-40, colors=colors)
        wedges, texts = ax[3].pie(gen_test_sizes, wedgeprops=dict(width=0.5), startangle=-40, colors=colors)
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        kw = dict(arrowprops=dict(arrowstyle="-"),
                bbox=bbox_props, zorder=0, va="center")

        plt.savefig(os.path.join(current_file_path,'../Rulebased_manual_gating/donut_plots/{0:}/Donut_plot_Pre_1hr_gen_{0:}_all_models.pdf'.format(patient_id)), format='pdf', dpi=600)
        plt.close()
    
    os.makedirs(os.path.join(current_file_path,'../data/gates/donut_plots/'), exist_ok=True)
    df_all_sizes.to_csv(os.path.join(current_file_path,'../data/gates/donut_plots/All_sizes.csv'), index=False, header=True)
    df_all_sizes_percent.to_csv(os.path.join(current_file_path,'../data/gates/donut_plots/All_sizes_percent.csv'), index=False, header=True)
    df_all_sizes_diff.to_csv(os.path.join(current_file_path,'../data/gates/donut_plots/All_sizes_percent_diff.csv'), index=False, header=True)
        

if __name__ == "__main__":
    
    current_file_path = os.path.abspath(os.path.dirname(__file__))
    # build_pca_scaler_all_gt_Pre_1hr(current_file_path)    
    Donut_PCA_colored_by_celltype(current_file_path)






    
    
