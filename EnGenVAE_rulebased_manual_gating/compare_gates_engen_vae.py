import sys
import os
import pandas as pd 
import numpy as np 
import random
import re 
import seaborn as sns 
from matplotlib import pyplot as plt
from matplotlib import cm, ticker
import pickle
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, pearsonr


def box_plot_freq(current_file_path):

    filename_engen_vae = os.path.join(current_file_path,'../data/gates_engen_vae/comparison_plots/Features_med_freq_1hr_gt_vs_test_gen.csv')
    data1 = 'gt_1hr_all'
    data2 = 'gen_test_all_models'
    df = pd.read_csv(filename_engen_vae)
    df1 = df[df['data_type']==data1]
    df2 = df[df['data_type']==data2]
    filename_engen = os.path.join(current_file_path,'../data/gates/comparison_plots/Features_med_freq_1hr_gt_vs_test_gen.csv')
    data3 = 'gen_test_all_models'
    df_vae = pd.read_csv(filename_engen)
    df3 = df_vae[df_vae['data_type']==data3]

    cell_types = df.columns.values
    cell_types = [i.split('.')[-1] for i in cell_types if i.split('.')[0]=='freq']
    cell_types.remove('mononuclear_cells')
    
    fig, ax = plt.subplots(nrows=4, ncols=4, sharex=False, sharey=False, figsize=(5,5))
    fig.suptitle('freq {} vs {}'.format(data1, data2), fontsize=5)
    
    for idx, cell_t in enumerate(cell_types):
        min_edge = 0
        max_edge = max(df1['freq.{}'.format(cell_t)])
        bin_list = np.linspace(min_edge, max_edge, 20)
        ax[idx//4,idx%4].hist(list(df2['freq.{}'.format(cell_t)]), bins=bin_list, alpha=0.7, label='EnGenVAE'+data2, color='goldenrod')
        ax[idx//4,idx%4].hist(list(df3['freq.{}'.format(cell_t)]), bins=bin_list, alpha=0.7, label='EnGen'+data3, color='cornflowerblue')
        ax[idx//4,idx%4].hist(list(df1['freq.{}'.format(cell_t)]), bins=bin_list, alpha=0.7, label=data1, color='seagreen')
        ax[idx//4,idx%4].set_title(cell_t, fontsize=5)
        plt.setp(ax[idx//4,idx%4].get_xticklabels(), fontsize=3, rotation = 0, ha='right')
        plt.setp(ax[idx//4,idx%4].get_yticklabels(), fontsize=3, rotation = 45, ha='right')

        
    plt.axis('square')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1, hspace=1)
    os.makedirs(os.path.join(current_file_path,'../EnGenVAE_rulebased_manual_gating/comparison_plots/'), exist_ok=True)
    plt.savefig(os.path.join(current_file_path,'../EnGenVAE_rulebased_manual_gating/comparison_plots/Freq_boxplot_{}_vs_{}.pdf'.format(data1, data2)), format='pdf', dpi=600)
    
def box_plot_med(current_file_path):

    filename_engen_vae = os.path.join(current_file_path,'../data/gates_engen_vae/comparison_plots/Features_med_freq_1hr_gt_vs_test_gen.csv')
    data1 = 'gt_1hr_all'
    data2 = 'gen_test_all_models'
    df = pd.read_csv(filename_engen_vae)
    df1 = df[df['data_type']==data1]
    df2 = df[df['data_type']==data2]
    filename_engen = os.path.join(current_file_path,'../data/gates/comparison_plots/Features_med_freq_1hr_gt_vs_test_gen.csv')
    data3 = 'gen_test_all_models'
    df_vae = pd.read_csv(filename_engen)
    df3 = df_vae[df_vae['data_type']==data3]

    cell_types = df.columns.values
    cell_types = [i.split('.')[-1] for i in cell_types if i.split('.')[0]=='freq']
    cell_types.remove('mononuclear_cells')
    
    fig, ax = plt.subplots(nrows=4, ncols=4, sharex=False, sharey=False, figsize=(5,5))
    fig.suptitle('median marker experssion {} vs {}'.format(data1, data2), fontsize=5)

    markers = ['149Sm_CREB', '150Nd_STAT5', '151Eu_p38', '153Eu_STAT1', '154Sm_STAT3', '155Gd_S6', '159Tb_MAPKAPK2', '164Dy_IkB', '166Er_NFkB', '167Er_ERK', '168Er_pSTAT6', '113In_CD235ab_CD61', '115In_CD45', '143Nd_CD45RA', '139La_CD66',
                    '141Pr_CD7', '142Nd_CD19', '144Nd_CD11b', '145Nd_CD4', '146Nd_CD8a', '147Sm_CD11c', '148Nd_CD123', '156Gd_CD24', '157Gd_CD161', '158Gd_CD33', '165Ho_CD16', '169Tm_CD25', '170Er_CD3', '171Yb_CD27', '172Yb_CD15',
                    '173Yb_CCR2', '175Lu_CD14',	'176Lu_CD56', '160Gd_Tbet', '162Dy_FoxP3', '152Sm_TCRgd', '174Yb_HLADR']
    
    for idx, cell_t in enumerate(cell_types):
        
        cell_type_cols = ['med.{}.{}'.format(marker,cell_t) for marker in markers]
        df1_cell_type = df1[cell_type_cols]
        df2_cell_type = df2[cell_type_cols]
        df3_cell_type = df3[cell_type_cols]
        pca = PCA(n_components=1)
        pca.fit(df1_cell_type)
        pca1 = pca.transform(df1_cell_type).flatten()
        pca2 = pca.transform(df2_cell_type).flatten()
        pca3 = pca.transform(df3_cell_type).flatten()
        min_edge = min(pca1)
        min_edge = min(min_edge,min(pca2))
        max_edge = max(pca1)
        max_edge = max(max_edge,max(pca2))
        bin_list = np.linspace(min_edge, max_edge, 20)
        ax[idx//4,idx%4].hist(pca2, bins=bin_list, alpha=0.7, label='EnGenVAE'+data2, color='goldenrod')
        ax[idx//4,idx%4].hist(pca3, bins=bin_list, alpha=0.7, label='EnGen'+data3, color='cornflowerblue')
        ax[idx//4,idx%4].hist(pca1, bins=bin_list, alpha=0.7, label=data1, color='seagreen')
        ax[idx//4,idx%4].set_title(cell_t, fontsize=5)
        plt.setp(ax[idx//4,idx%4].get_xticklabels(), fontsize=3, rotation = 0, ha='right')
        plt.setp(ax[idx//4,idx%4].get_yticklabels(), fontsize=3, rotation = 45, ha='right')
  
    plt.axis('square')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1, hspace=1)
    os.makedirs(os.path.join(current_file_path,'../EnGenVAE_rulebased_manual_gating/comparison_plots/'), exist_ok=True)
    plt.savefig(os.path.join(current_file_path,'../EnGenVAE_rulebased_manual_gating/comparison_plots/Med_boxplot_{}_vs_{}.pdf'.format(data1, data2)), format='pdf', dpi=600)
    

if __name__ == "__main__":
    

    current_file_path = os.path.abspath(os.path.dirname(__file__))

    box_plot_freq(current_file_path)
    box_plot_med(current_file_path)
    






    
    
