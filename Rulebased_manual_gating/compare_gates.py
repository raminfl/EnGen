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


def compare_freq(current_file_path):

    filename = os.path.join(current_file_path,'../data/gates/comparison_plots/Features_med_freq_1hr_gt_vs_test_gen.csv')
    data1 = 'gt_1hr_all'
    data2 = 'gen_test_all_models'
    df = pd.read_csv(filename)
    df1 = df[df['data_type']==data1]
    df2 = df[df['data_type']==data2]
    df1.sort_values(by=['patient_id'], inplace=True)
    df2.sort_values(by=['patient_id'], inplace=True)
    cell_types = df.columns.values
    cell_types = [i.split('.')[-1] for i in cell_types if i.split('.')[0]=='freq']
    cell_types.remove('mononuclear_cells')
    
    fig, ax = plt.subplots(nrows=4, ncols=4, sharex=False, sharey=False, figsize=(5,5))
    fig.suptitle('freq {} vs {}'.format(data1, data2), fontsize=5)
    df_corrs = pd.DataFrame(columns=['cell_type', 'R', 'pval'])
    for idx, cell_t in enumerate(cell_types):

        sns.regplot(x=list(df1['freq.{}'.format(cell_t)]), y=list(df2['freq.{}'.format(cell_t)]), ax=ax[idx//4,idx%4], robust=True, color='#2d3436', x_ci='ci', ci=95, line_kws={'linewidth':1, 'color':'black'}, scatter_kws={'color':'grey', 's':3})
        ax[idx//4,idx%4].set_title(cell_t, fontsize=5)
        plt.setp(ax[idx//4,idx%4].get_xticklabels(), fontsize=3, rotation = 0, ha='right')
        plt.setp(ax[idx//4,idx%4].get_yticklabels(), fontsize=3, rotation = 45, ha='right')
        ax[idx//4,idx%4].set_xlim(min(min(df1['freq.{}'.format(cell_t)]),min(df2['freq.{}'.format(cell_t)])),max(max(df1['freq.{}'.format(cell_t)]),max(df2['freq.{}'.format(cell_t)])))
        ax[idx//4,idx%4].set_ylim(min(min(df1['freq.{}'.format(cell_t)]),min(df2['freq.{}'.format(cell_t)])),max(max(df1['freq.{}'.format(cell_t)]),max(df2['freq.{}'.format(cell_t)])))
        r, pval = pearsonr(df1['freq.{}'.format(cell_t)], df2['freq.{}'.format(cell_t)])
        df_corrs.loc[idx,:] = [cell_t, r, pval]

    plt.axis('square')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1, hspace=1)
    df_corrs.to_csv(os.path.join(current_file_path,'../data/gates/comparison_plots/Freq_comparison_pearson_{}_vs_{}.csv'.format(data1, data2)), header=True, index=False)
    os.makedirs(os.path.join(current_file_path,'../Rulebased_manual_gating/comparison_plots/'), exist_ok=True)
    plt.savefig(os.path.join(current_file_path,'../Rulebased_manual_gating/comparison_plots/Freq_comparison_{}_vs_{}.pdf'.format(data1, data2)), format='pdf', dpi=600)
    
    
    
    
def compare_med(current_file_path):
    # compare only the first PC
    filename = os.path.join(current_file_path,'../data/gates/comparison_plots/Features_med_freq_1hr_gt_vs_test_gen.csv')
    data1 = 'gt_1hr_all'
    data2 = 'gen_test_all_models'
    df = pd.read_csv(filename)
    df1 = df[df['data_type']==data1]
    df2 = df[df['data_type']==data2]
    df1.sort_values(by=['patient_id'], inplace=True)
    df2.sort_values(by=['patient_id'], inplace=True)
    cell_types = df.columns.values
    cell_types = [i.split('.')[-1] for i in cell_types if i.split('.')[0]=='freq']
    cell_types.remove('mononuclear_cells')

    markers = ['149Sm_CREB', '150Nd_STAT5', '151Eu_p38', '153Eu_STAT1', '154Sm_STAT3', '155Gd_S6', '159Tb_MAPKAPK2', '164Dy_IkB', '166Er_NFkB', '167Er_ERK', '168Er_pSTAT6', '113In_CD235ab_CD61', '115In_CD45', '143Nd_CD45RA', '139La_CD66',
                    '141Pr_CD7', '142Nd_CD19', '144Nd_CD11b', '145Nd_CD4', '146Nd_CD8a', '147Sm_CD11c', '148Nd_CD123', '156Gd_CD24', '157Gd_CD161', '158Gd_CD33', '165Ho_CD16', '169Tm_CD25', '170Er_CD3', '171Yb_CD27', '172Yb_CD15',
                    '173Yb_CCR2', '175Lu_CD14',	'176Lu_CD56', '160Gd_Tbet', '162Dy_FoxP3', '152Sm_TCRgd', '174Yb_HLADR']

    fig, ax = plt.subplots(nrows=4, ncols=4, sharex=False, sharey=False, figsize=(5,5))
    fig.suptitle('median marker experssion {} vs {}'.format(data1, data2), fontsize=5)
    df_corrs = pd.DataFrame(columns=['cell_type', 'R', 'pval'])
    for idx, cell_t in enumerate(cell_types):
        print('************************* {}'.format(cell_t))
        cell_type_cols = ['med.{}.{}'.format(marker,cell_t) for marker in markers]
        #print(cell_type_cols)
        df1_cell_type = df1[cell_type_cols]
        df2_cell_type = df2[cell_type_cols]
        pca = PCA(n_components=1)
        pca.fit(df1_cell_type)
        pca1 = pca.transform(df1_cell_type)
        pca2 = pca.transform(df2_cell_type)
        sns.regplot(x=pca1, y=pca2, ax=ax[idx//4,idx%4], robust=True, color='black', x_ci='ci', ci=95, line_kws={'linewidth':1, 'color':'black'}, scatter_kws={'color':'#2d3436', 's':3})
        ax[idx//4,idx%4].set_title(cell_t, fontsize=5)
        plt.setp(ax[idx//4,idx%4].get_xticklabels(), fontsize=3, rotation = 0, ha='right')
        plt.setp(ax[idx//4,idx%4].get_yticklabels(), fontsize=3, rotation = 45, ha='right')
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1, hspace=1)
        ax[idx//4,idx%4].set_xlim(min(min(pca1),min(pca2)),max(max(pca1),max(pca2)))
        ax[idx//4,idx%4].set_ylim(min(min(pca1),min(pca2)),max(max(pca1),max(pca2)))
        r, pval = pearsonr(pca1.flatten(), pca2.flatten())
        df_corrs.loc[idx,:] = [cell_t, r, pval]
  
    df_corrs.to_csv(os.path.join(current_file_path,'../data/gates/comparison_plots/Med_comparison_pearson_{}_vs_{}.csv'.format(data1, data2)), header=True, index=False)
    plt.savefig(os.path.join(current_file_path,'../Rulebased_manual_gating/comparison_plots/Med_comparison_{}_vs_{}.pdf'.format(data1, data2)), format='pdf', dpi=600)

        
def compare_med_each_marker(current_file_path):
    # do not reduce to 1st pc

    filename = os.path.join(current_file_path,'../data/gates/comparison_plots/Features_med_freq_1hr_gt_vs_test_gen.csv')
    data1 = 'gt_1hr_all'
    data2 = 'gen_test_all_models'
    df = pd.read_csv(filename)
    df1 = df[df['data_type']==data1]
    df2 = df[df['data_type']==data2]
    df1.sort_values(by=['patient_id'], inplace=True)
    df2.sort_values(by=['patient_id'], inplace=True)
    cell_types = df.columns.values
    cell_types = [i.split('.')[-1] for i in cell_types if i.split('.')[0]=='freq']
    temp_pops = ['mononuclear_cells', 'Tcells_CD8pos', 'Tcells_CD4pos', 'Tcells_CD4negCD8neg', 'Tcells', 'CD19negCD3neg', 'Bcells']
    cell_types = [i for i in cell_types if i not in temp_pops]

    markers = ['149Sm_CREB', '150Nd_STAT5', '151Eu_p38', '153Eu_STAT1', '154Sm_STAT3', '155Gd_S6', '159Tb_MAPKAPK2', '164Dy_IkB', '166Er_NFkB', '167Er_ERK', '168Er_pSTAT6', '113In_CD235ab_CD61', '115In_CD45', '143Nd_CD45RA', '139La_CD66',
                    '141Pr_CD7', '142Nd_CD19', '144Nd_CD11b', '145Nd_CD4', '146Nd_CD8a', '147Sm_CD11c', '148Nd_CD123', '156Gd_CD24', '157Gd_CD161', '158Gd_CD33', '165Ho_CD16', '169Tm_CD25', '170Er_CD3', '171Yb_CD27', '172Yb_CD15',
                    '173Yb_CCR2', '175Lu_CD14',	'176Lu_CD56', '160Gd_Tbet', '162Dy_FoxP3', '152Sm_TCRgd', '174Yb_HLADR']

    df_corrs = pd.DataFrame(columns=['cell_type', 'marker', 'R', 'pval'])
    for idx1, cell_t in enumerate(cell_types):
        print('************************* {}'.format(cell_t))
        for idx2, marker in enumerate(markers):
            #print('************************* {}'.format(marker))
            cell_type_cols = 'med.{}.{}'.format(marker,cell_t)
            df1_cell_type = df1[cell_type_cols]
            df2_cell_type = df2[cell_type_cols]
            r, pval = pearsonr(df1_cell_type, df2_cell_type)
            df_corrs.loc[df_corrs.shape[0],:] = [cell_t, marker, r, -1*np.log10(pval)]
            
    df_corrs.to_csv(os.path.join(current_file_path,'../data/gates/comparison_plots/Med_comparison_pearson_{}_vs_{}_each_marker.csv'.format(data1, data2)), header=True, index=False)
    g = sns.relplot(data=df_corrs, x='marker', y='cell_type', hue='R', size='pval', palette='RdBu', edgecolor=".7", height=10, sizes=(0,150), size_norm=(0,20), hue_norm=(-1, 1))
    # Tweak the figure to finalize
    g.set(xlabel="", ylabel="", aspect="equal")
    g.despine(left=True, bottom=True)
    g.ax.margins(.02)
    for label in g.ax.get_xticklabels():
        label.set_rotation(90)
    plt.savefig(os.path.join(current_file_path,'../Rulebased_manual_gating/comparison_plots/Relplot_med_comparison_{}_vs_{}.pdf'.format(data1, data2)), format='pdf', dpi=600)


def plot_corr_coeff_pvals(current_file_path):
    # for fig 4B

    data1 = 'gt_1hr_all'
    data2 = 'gen_test_all_models'
    df_corrs_med = pd.read_csv(os.path.join(current_file_path,'../data/gates/comparison_plots/Med_comparison_pearson_{}_vs_{}.csv'.format(data1, data2)))
    df_corrs_freq = pd.read_csv(os.path.join(current_file_path,'../data/gates/comparison_plots/Freq_comparison_pearson_{}_vs_{}.csv'.format(data1, data2)))
    df_corrs_med['log_pval'] = -1*np.log10(df_corrs_med['pval'])
    df_corrs_freq['log_pval'] = -1*np.log10(df_corrs_freq['pval'])
    df_corrs = pd.DataFrame()
    assert (df_corrs_freq['cell_type']==df_corrs_med['cell_type']).all, 'indeces do not match!'
    df_corrs['med_R'] = df_corrs_med['R']
    df_corrs['freq_R'] = df_corrs_freq['R']
    df_corrs['med_logpval'] = df_corrs_med['log_pval']
    df_corrs['freq_logpval'] = df_corrs_freq['log_pval']
    df_corrs.index = df_corrs_med['cell_type']
    # print(df_corrs)
    pops = ['Bcells_Mem', 'Bcells_Naive', 'NKcellsCD7pos', 'MCcells', 'TCRgd', 'CD4posMem', 'CD4posNaive', 'CD8posMem', 'CD8posNaive']
    pops = pops[::-1]
    keep_indx = [True if i in pops else False for i in df_corrs_med['cell_type']]
    df_corrs = df_corrs[keep_indx]
    # print(df_corrs)
    df_corrs = df_corrs.loc[pops] # sort rows by the order in pops
    
    fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(15,5))
    ax[0].spines['right'].set_zorder(10)
    ax[1].spines['left'].set_zorder(10)
    df_corrs[['med_R','freq_R']].plot.barh(ax=ax[0], width=0.6, color=['#FBC56C','#8EC7FA'], edgecolor='white', linewidth=0.1, legend=False, zorder=3)
    ax[0].set(title='Correlation Coefficient')
    df_corrs[['med_logpval','freq_logpval']].plot.barh(ax=ax[1], color=['#FBC56C','#8EC7FA'], width=0.6, edgecolor='white', linewidth=0.1, legend=False, zorder=3)
    ax[1].set(title='P value')

    ax[1].vlines(-1*np.log10(0.05), -10, 10, zorder=5, color='#006400', linewidth=4)

    ax[0].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax[0].grid(True, axis='x', zorder=0, linestyle='--')
    ax[0].set_xlim((0,1))

    ax[1].set_xticks([0, 4, 8, 12, 16, 20])
    ax[1].set_xticklabels(['', '4', '8', '12', '16', '20'])
    ax[1].grid(True, axis='x', zorder=0, linestyle='--')
    ax[1].set_xlim((0,20))

    ax[0].invert_xaxis()
    ax[0].yaxis.tick_right()

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.3)
        
    plt.savefig(os.path.join(current_file_path,'../Rulebased_manual_gating/comparison_plots/Corr_pvals_plot_{}_vs_{}.pdf'.format(data1, data2)), format='pdf', dpi=600)


if __name__ == "__main__":
    

    current_file_path = os.path.abspath(os.path.dirname(__file__))

    compare_freq(current_file_path)
    compare_med(current_file_path)
    compare_med_each_marker(current_file_path)
    plot_corr_coeff_pvals(current_file_path)







    
    
