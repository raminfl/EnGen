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
from sklearn.svm import SVC
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, auc, plot_confusion_matrix
from multiprocessing import Process
from itertools import repeat
from multiprocessing import Pool
import seaborn as sns








def boxplots_RF_test_gen_test(score, fscores_path, save_path, folder_name, n_clusters):

  
    df_fscores = pd.read_csv(fscores_path+folder_name+'/RF_scores_{0:}_clusters.csv'.format(n_clusters))
   
    print(df_fscores)
    fig, ax = plt.subplots(figsize=(3,4.5))
    
    
    ax = sns.boxplot(y=score, data=df_fscores, showfliers=False, linewidth=3, color='#4d908e')
    l = ax.lines[4] # median line
    # l.set_linestyle(':')
    l.set_color('black')
    l.set_linewidth(6)

    ax = sns.swarmplot(y=score, data=df_fscores, size=4, color='#DEE2E6', edgecolor="black", linewidth=1.0)

    if score == 'accuracy_score_test_st':
        plt.title('Accuracy 50% test GT\nn_clusters = {}'.format(n_clusters))
    elif score == 'accuracy_score_gen':
        plt.title('Accuracy Gen (labeled 1hr)\nn_clusters = {}'.format(n_clusters))
    elif score == 'auc_test_st':
        plt.title('AUC 50% test GT\nn_clusters = {}'.format(n_clusters))
    elif score == 'auc_test_1hr':
        plt.title('AUC GT 1hr + GEN(labeled Pre)\nn_clusters = {}'.format(n_clusters))
    else:
        print('score = {} not defined'.format(score))
        sys.exit()
    plt.ylim(0,1.1)
    # plt.show()
    plt.savefig('/home/raminf/DL/FCS_files_kehlet/iter_33pc_train/Classification_results_test_gen/run_0/box_plot_{0:}_{1:}clusters.pdf'.format(score, n_clusters), format='pdf', dpi=600)

def boxplots_RF_test_gen_test_fig5C(score, fscores_path, save_path, folder_name, n_clusters):

  
    df_fscores = pd.read_csv(fscores_path+folder_name+'/RF_scores_{0:}_clusters.csv'.format(n_clusters))
   
    print(df_fscores)
    fig, ax = plt.subplots(figsize=(7,2.5))
    
    
    ax = sns.boxplot(y=score, data=df_fscores, showfliers=False, linewidth=3, color='#dfe6e9', orient='h')
    l = ax.lines[4] # median line
    # l.set_linestyle(':')
    l.set_color('black')
    l.set_linewidth(3)

    ax = sns.swarmplot(y=score, data=df_fscores, size=4, color='#DEE2E6', edgecolor="black", linewidth=1.0, orient='h')

    if score == 'accuracy_score_test_st':
        plt.title('Accuracy 50% test GT\nn_clusters = {}'.format(n_clusters))
    elif score == 'accuracy_score_gen':
        plt.title('Accuracy Gen (labeled 1hr)\nn_clusters = {}'.format(n_clusters))
    elif score == 'auc_test_st':
        plt.title('AUC 50% test GT\nn_clusters = {}'.format(n_clusters))
    elif score == 'auc_test_1hr':
        plt.title('AUC GT 1hr + GEN(labeled Pre)\nn_clusters = {}'.format(n_clusters))
    else:
        print('score = {} not defined'.format(score))
        sys.exit()
    plt.xlim(0,1)
    # plt.show()
    plt.savefig(save_path+folder_name+'/box_plot_{0:}_{1:}clusters_vertical.pdf'.format(score, n_clusters), format='pdf', dpi=600)


def boxplots_RF_baseline_performance(fscores_path, save_path, folder_name, n_clusters):

    # plot baseline performance of the RF classifiers
    df_fscores = pd.read_csv(fscores_path+folder_name+'/RF_scores_{0:}_clusters.csv'.format(n_clusters))
   
    print(df_fscores)
    
    
    for i, score in enumerate(['accuracy_score_test_st', 'auc_test_st', 'accuracy_score_gen']):
        fig, ax = plt.subplots(figsize=(2.5, 4))
        ax = sns.boxplot(y=score, data=df_fscores, showfliers=False, linewidth=3, color='#dfe6e9')
        l = ax.lines[4] # median line
        # l.set_linestyle(':')
        l.set_color('black')
        l.set_linewidth(3)

        ax = sns.swarmplot(y=score, data=df_fscores, size=4, color='#DEE2E6', edgecolor="black", linewidth=1.0)

        # if score == 'accuracy_score_test_st':
        #     ax.set_title('Accuracy 50% test GT\nn_clusters = {}'.format(n_clusters))
        # elif score == 'accuracy_score_gen':
        #     ax.set_title('Accuracy Gen (labeled 1hr)\nn_clusters = {}'.format(n_clusters))
        # elif score == 'auc_test_st':
        #     ax.set_title('AUC 50% test GT\nn_clusters = {}'.format(n_clusters))
        # elif score == 'auc_test_1hr':
        #     ax.set_title('AUC GT 1hr + GEN(labeled Pre)\nn_clusters = {}'.format(n_clusters))
        # else:
        #     print('score = {} not defined'.format(score))
        #     sys.exit()
        ax.set_ylim(0,1)
        plt.tight_layout()
        # plt.show()
        plt.savefig(save_path+folder_name+'/box_plot_baseline_RF_performance_{0:}_{1:}clusters.pdf'.format(score, n_clusters), format='pdf', dpi=600)

   

if __name__ == "__main__":

    
    folder_name = 'run_0'

    n_clusters = 25
    
    fscores_path = '/home/raminf/DL/EnGen/FCS_files_kehlet/iter_33pc_train/Classification_results_test_gen/'
    save_path = '/home/raminf/DL/EnGen/FCS_files_kehlet/iter_33pc_train/Classification_results_test_gen/'
    # score = 'accuracy_score_test_st'
    # score = 'accuracy_score_gen'
    # score = 'auc_test_st'
    # score = 'auc_test_1hr'
    # boxplots_RF_test_gen_test(score, fscores_path, save_path, folder_name, n_clusters)
    # boxplots_RF_test_gen_test_fig5C(score, fscores_path, save_path, folder_name, n_clusters)
    boxplots_RF_baseline_performance(fscores_path, save_path, folder_name, n_clusters)

        