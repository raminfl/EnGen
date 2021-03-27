import sys
import os
import pandas as pd 
import numpy as np 
import random
import seaborn as sns 
from matplotlib import pyplot as plt
from matplotlib import cm
import itertools







def boxplots_RF_test_gen_test(scores_path, n_clusters=25):
    # AUC of trained RF models when fed gt Post and gen Post (but labeled as 'Pre')
  
    df_scores = pd.read_csv(scores_path+'/RF_scores_{0:}_clusters.csv'.format(n_clusters))

    score = 'auc_test_1hr'
    fig, ax = plt.subplots(figsize=(7,2.5))
    ax = sns.boxplot(y=score, data=df_scores, showfliers=False, linewidth=3, color='#dfe6e9', orient='h')
    l = ax.lines[4] # median line
    l.set_color('black')
    l.set_linewidth(3)
    ax = sns.swarmplot(y=score, data=df_scores, size=4, color='#DEE2E6', edgecolor="black", linewidth=1.0, orient='h')
    plt.title('AUC GT 1hr + GEN(labeled Pre)\nn_clusters = {}'.format(n_clusters))
    plt.xlim(0,1)
    plt.savefig(scores_path+'/box_plot_{0:}_{1:}clusters.pdf'.format(score, n_clusters), format='pdf', dpi=600)


def boxplots_RF_baseline_performance(scores_path, n_clusters=25):

    # plot baseline performance of the RF classifiers
    df_scores = pd.read_csv(scores_path+'/RF_scores_{0:}_clusters.csv'.format(n_clusters))
    
    for i, score in enumerate(['auc_test_st', 'accuracy_score_gen']):
        fig, ax = plt.subplots(figsize=(2.5, 4))
        ax = sns.boxplot(y=score, data=df_scores, showfliers=False, linewidth=3, color='#dfe6e9')
        l = ax.lines[4] # median line
        # l.set_linestyle(':')
        l.set_color('black')
        l.set_linewidth(3)
        ax = sns.swarmplot(y=score, data=df_scores, size=4, color='#DEE2E6', edgecolor="black", linewidth=1.0)
        ax.set_ylim(0,1)
        plt.tight_layout()
        plt.savefig(scores_path+'box_plot_baseline_RF_performance_{0:}_{1:}clusters.pdf'.format(score, n_clusters), format='pdf', dpi=600)

   

if __name__ == "__main__":
    
    current_file_path = os.path.abspath(os.path.dirname(__file__))
    scores_path = os.path.join(current_file_path,'../EnGen_train_iterations/engen_output/Classification_results/')

    boxplots_RF_test_gen_test(scores_path)
    boxplots_RF_baseline_performance(scores_path)

        