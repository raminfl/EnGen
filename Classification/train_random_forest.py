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
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, auc, plot_confusion_matrix, accuracy_score
from multiprocessing import Process
from itertools import repeat
from multiprocessing import Pool


def random_forrest_classifier_train_gt_test_mix_one_model(feat_path, save_path, folder_name, iter_id, n_clusters):

    random.seed(42+iter_id)
    print(iter_id)
    if iter_id < 10:
        filename = '/home/raminf/DL/EnGen/FCS_files_kehlet/iter_33pc_train/plots/PCA_Pre_1hr/iter_0{0:}/Features_gt_Pre_gt_1hr_gen_1hr_test_train_kmeans{1:}_iter_0{0:}.csv'.format(iter_id, n_clusters)
        p = pickle.load(open('/home/raminf/DL/EnGen/FCS_files_kehlet/iter_33pc_train/preprocessed_data/run_0/iter_0{0:}/scaler_kmeans_25.pkl'.format(iter_id), 'rb'))
    else:
        filename = '/home/raminf/DL/EnGen/FCS_files_kehlet/iter_33pc_train/plots/PCA_Pre_1hr/iter_{0:}/Features_gt_Pre_gt_1hr_gen_1hr_test_train_kmeans{1:}_iter_{0:}.csv'.format(iter_id, n_clusters)
        p = pickle.load(open('/home/raminf/DL/EnGen/FCS_files_kehlet/iter_33pc_train/preprocessed_data/run_0/iter_{0:}/scaler_kmeans_25.pkl'.format(iter_id), 'rb'))

    AE_train_ids = p['AE_train_ids']

    df_featues = pd.read_csv(filename)
    # df_featues['patient_id'] = [str(i) for i in df_featues['patient_id']]
    # print(df_featues)
    df_featues['timepoint'] = label_binarize(df_featues['timepoint'], classes=['Pre', '1hr']).flatten()
    
    # print(df_featues)
    for AE_train_id in AE_train_ids:
        df_featues = df_featues[df_featues['patient_id']!=AE_train_id] # removed patients used in AE training. both gt and gen.
    df_gt = df_featues[df_featues['data_type']=='ground_truth'] # features from gt files
    df_gt.reset_index(inplace=True, drop=True)
    
    

    df_gen = df_featues[df_featues['data_type']=='gen_test'] # features from generated test files
    df_gen.reset_index(inplace=True, drop=True)
    # print(df_gt)
    # print(df_gen)

    # scaler = StandardScaler()
    # scaler.fit(df_gt_train.iloc[:, 3:])
    # df_gt_train.iloc[:, 3:] = scaler.transform(df_gt_train.iloc[:, 3:].values)     
    # df_gt_test.iloc[:, 3:] = scaler.transform(df_gt_test.iloc[:, 3:].values)  
    # df_gen_test.iloc[:, 3:] = scaler.transform(df_gen_test.iloc[:, 3:].values)  
    
    X = df_gt.iloc[:, 3:].values
    y = df_gt['timepoint'].values

    
    
    



    # clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=50)
    # clf.fit(X, y)

    # clf = RandomForestClassifier(random_state=42, n_jobs=5)
    # specify parameters and distributions to sample from
    # param_dist = {'n_estimators': [20, 50, 100, 200, 500],
    #           'criterion': ['gini', 'entropy'],
    #           'max_depth': [None, 2, 5, 10],
    #           'min_samples_split': [2, 5, 10],
    #           'max_features': ['auto', 'sqrt', 'log2']}
              
    # clf = SVC(random_state=42, probability=True)
    # # specify parameters and distributions to sample from
    # param_dist = {'C': np.logspace(-2,1,60),
    #           'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    #           'degree': [2, 3, 4]}

    # run randomized search
    # n_iter_search = 200
    
    skf = StratifiedKFold(n_splits=2, shuffle=False)
    fold = -1
    for train_index, test_index in skf.split(X, y):
        fold += 1
        # print("TRAIN:", train_index, "TEST:", test_index)
    
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        df_gt_Pre_train = df_gt.iloc[train_index,:] # features from gt files from Pre used to train RF
        # print(df_gt_Pre_train)
        df_gt_Pre_train = df_gt_Pre_train[df_gt_Pre_train['timepoint']==0] # Pre
        # print(df_gt_Pre_train)
        discard_gen_patient_ids = df_gt_Pre_train['patient_id'].values
        # print(discard_gen_patient_ids)
        df_gt_Pre_train.reset_index(inplace=True, drop=True)
        X_Pre_train = df_gt_Pre_train.iloc[:, 3:].values
        y_Pre_train = df_gt_Pre_train['timepoint'].values

        df_gt_Pre_test = df_gt.iloc[test_index,:] # features from gt files from Pre
        df_gt_Pre_test = df_gt_Pre_test[df_gt_Pre_test['timepoint']==0] # Pre
        df_gt_Pre_test.reset_index(inplace=True, drop=True)
        X_Pre_test = df_gt_Pre_test.iloc[:, 3:].values
        y_Pre_test = df_gt_Pre_test['timepoint'].values

        df_gt_1hr_test = df_gt.iloc[test_index,:] # features from gt files from Pre
        df_gt_1hr_test = df_gt_1hr_test[df_gt_1hr_test['timepoint']==1] # 1hr
        df_gt_1hr_test.reset_index(inplace=True, drop=True)
        X_1hr_test = df_gt_1hr_test.iloc[:, 3:].values
        y_1hr_test = df_gt_1hr_test['timepoint'].values

        df_gen_fold = df_gen.copy()
        for gen_patient in discard_gen_patient_ids:
            df_gen_fold = df_gen_fold[df_gen_fold['patient_id']!=gen_patient] # removed gen patients that their gt was used in training of this fold
        # print(df_gen_fold)
        X_gen = df_gen_fold.iloc[:, 3:].values
        y_gen = df_gen_fold['timepoint'].values

        y_1hr = [1 for _ in range(df_gt_Pre_test.shape[0])]

        clf = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=10)
        clf.fit(X_train, y_train)


        # y_pred_gt = clf.predict(X_test_gt)
        # y_pred_gt_prob = clf.predict_proba(X_test_gt)
        # y_test_gt_binarized = label_binarize(y_test_gt, classes=clf.classes_)
        y_pred_train_st = clf.predict(X_train)
        y_pred_train_st_prob = clf.predict_proba(X_train)

        y_pred_test_st = clf.predict(X_test)
        y_pred_test_st_prob = clf.predict_proba(X_test)
        # y_test_binarized = label_binarize(y_test, classes=clf.classes_)

        y_pred_train_Pre = clf.predict(X_Pre_train)
        y_pred_test_Pre = clf.predict(X_Pre_test)

        y_pred_test_1hr = clf.predict(X_1hr_test)
        y_pred_test_1hr_prob = clf.predict_proba(X_1hr_test)

        y_pred_gen = clf.predict(X_gen)
        y_pred_gen_prob = clf.predict_proba(X_gen)
    
        # print('---------- Prediction of both Pre and 1hr on 50% RF Train')
        # print(y_train)
        # print(y_pred_train_st)
        # print('---------- Prediction of both Pre and 1hr on 50% RF Test')
        # print(y_test)
        # print(y_pred_test_st)
        # print('---------- Prediction of Pre on 50% RF Train')
        # print(y_Pre_train)
        # print(y_pred_train_Pre)
        # print('---------- Prediction of Pre on 50% RF Test')
        # print(y_Pre_test)
        # print(y_pred_test_Pre)
        # print('---------- Prediction of 1hr on 50% RF Test')
        # print(y_gen)
        # print(y_pred_gen)
        # print('----------')

        accuracy_score_train_st = accuracy_score(y_train, y_pred_train_st)
        accuracy_score_test_st = accuracy_score(y_test, y_pred_test_st)
        # accuracy_score_train_A = accuracy_score(y_gen, y_pred_gen) # accuracy of model when input of AE is fed
        # accuracy_score_test_A = accuracy_score(y_B, y_pred_test_A) # accuracy of model when input of AE is fed
        accuracy_score_gen = accuracy_score(y_gen, y_pred_gen) # accuracy of model when output of AE is fed
        # print(y_pred_train_st)
        # print(y_train)
        # print(roc_auc_score(y_train, y_pred_train_st_prob[:,1]))
        # print(y_pred_train_st_prob[:,1])
        auc_train_st = roc_auc_score(y_train, y_pred_train_st_prob[:,1])
        # print(roc_auc_score(y_test, y_pred_test_st_prob[:,1]))
        auc_test_st = roc_auc_score(y_test, y_pred_test_st_prob[:,1])

        y_mix = np.concatenate((y_1hr_test, np.asarray([0 for _ in range(len(y_gen))])), axis=0)  # mix 1hr RF test 1hr gt with y_gen_test (y_gen_test is excluding AE and RF train patients resulting in 9 samples per label, 18 total)
        y_mix_pred = np.concatenate((y_pred_test_1hr_prob[:,1], y_pred_gen_prob[:,1]), axis=0)
        # print(y_mix)
        # print(y_mix_pred)

        # print(roc_auc_score(y_mix, y_mix_pred))
        auc_test_1hr = roc_auc_score(y_mix, y_mix_pred)
        # print(auc_test_st)

        df_RF_scores = pd.DataFrame(data=[[iter_id, fold, accuracy_score_train_st, accuracy_score_test_st, accuracy_score_gen, auc_train_st, auc_test_st, auc_test_1hr]], 
            columns=['iter_id', 'fold', 'accuracy_score_train_st', 'accuracy_score_test_st', 'accuracy_score_gen', 'auc_train_st', 'auc_test_st', 'auc_test_1hr'])
        
        if os.path.exists(save_path+folder_name+'/RF_scores_{}_clusters.csv'.format(n_clusters)):
            df_RF_scores.to_csv(save_path+folder_name+'/RF_scores_{}_clusters.csv'.format(n_clusters), mode='a', index=False, header=False)
        else:
            os.makedirs(save_path+folder_name, exist_ok=True)
            df_RF_scores.to_csv(save_path+folder_name+'/RF_scores_{}_clusters.csv'.format(n_clusters), index=False, header=True)
        


if __name__ == "__main__":
    
    folder_name = 'run_0'

    feat_path = '/home/raminf/DL/EnGen/FCS_files_kehlet/iter_33pc_train/plots/PCA_Pre_1hr/'
    save_path = '/home/raminf/DL/EnGen/FCS_files_kehlet/iter_33pc_train/Classification_results_test_gen/'

    # n_clusters = 10
    n_clusters = 25
   
    for iter_id in range(30):
        random_forrest_classifier_train_gt_test_mix_one_model(feat_path, save_path, folder_name, iter_id, n_clusters)