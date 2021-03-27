import sys
import os
import pandas as pd 
import numpy as np 
import random
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score


def random_forrest_classifier_train(current_file_path, iter_id, n_clusters=25):

    random.seed(42+iter_id)

    if iter_id < 10:
        filename = os.path.join(current_file_path,'../EnGen_train_iterations/engen_output/iter_0{0:}/Features_gt_Pre_gt_1hr_gen_1hr_test_train_kmeans{1:}_iter_0{0:}.csv'.format(iter_id, n_clusters))
        p = pickle.load(open(os.path.join(current_file_path,'../EnGen_train_iterations/training_data/iter_0{0:}/scaler_kmeans_25.pkl'.format(iter_id)), 'rb'))
    else:
        filename = os.path.join(current_file_path,'../EnGen_train_iterations/engen_output/iter_{0:}/Features_gt_Pre_gt_1hr_gen_1hr_test_train_kmeans{1:}_iter_{0:}.csv'.format(iter_id, n_clusters))
        p = pickle.load(open(os.path.join(current_file_path,'../EnGen_train_iterations/training_data/iter_{0:}/scaler_kmeans_25.pkl'.format(iter_id)), 'rb'))

    AE_train_ids = p['AE_train_ids']

    df_featues = pd.read_csv(filename)
    df_featues['timepoint'] = label_binarize(df_featues['timepoint'], classes=['Pre', '1hr']).flatten()

    for AE_train_id in AE_train_ids:
        df_featues = df_featues[df_featues['patient_id']!=AE_train_id] # removed patients used in AE training. both gt and gen.
    df_gt = df_featues[df_featues['data_type']=='ground_truth'] # features from gt files
    df_gt.reset_index(inplace=True, drop=True)
    df_gen = df_featues[df_featues['data_type']=='gen_test'] # features from generated test files
    df_gen.reset_index(inplace=True, drop=True)
    X = df_gt.iloc[:, 3:].values
    y = df_gt['timepoint'].values    
    skf = StratifiedKFold(n_splits=2, shuffle=False)
    fold = -1
    for train_index, test_index in skf.split(X, y):
        fold += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        df_gt_Pre_train = df_gt.iloc[train_index,:] # features from gt files from Pre used to train RF
        df_gt_Pre_train = df_gt_Pre_train[df_gt_Pre_train['timepoint']==0] # Pre
        discard_gen_patient_ids = df_gt_Pre_train['patient_id'].values
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
        
        X_gen = df_gen_fold.iloc[:, 3:].values
        y_gen = df_gen_fold['timepoint'].values

        y_1hr = [1 for _ in range(df_gt_Pre_test.shape[0])]

        clf = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=10)
        clf.fit(X_train, y_train)

        y_pred_train_st = clf.predict(X_train)
        y_pred_train_st_prob = clf.predict_proba(X_train)

        y_pred_test_st = clf.predict(X_test)
        y_pred_test_st_prob = clf.predict_proba(X_test)

        y_pred_train_Pre = clf.predict(X_Pre_train)
        y_pred_test_Pre = clf.predict(X_Pre_test)

        y_pred_test_1hr = clf.predict(X_1hr_test)
        y_pred_test_1hr_prob = clf.predict_proba(X_1hr_test)

        y_pred_gen = clf.predict(X_gen)
        y_pred_gen_prob = clf.predict_proba(X_gen)

        accuracy_score_train_st = accuracy_score(y_train, y_pred_train_st)
        accuracy_score_test_st = accuracy_score(y_test, y_pred_test_st)

        accuracy_score_gen = accuracy_score(y_gen, y_pred_gen) # accuracy of model when output of AE is fed

        auc_train_st = roc_auc_score(y_train, y_pred_train_st_prob[:,1])
        auc_test_st = roc_auc_score(y_test, y_pred_test_st_prob[:,1])

        y_mix = np.concatenate((y_1hr_test, np.asarray([0 for _ in range(len(y_gen))])), axis=0)  # mix RF test 1hr gt with y_gen_test (y_gen_test is excluding AE and RF train patients resulting in 9 samples per label, 18 total)
        y_mix_pred = np.concatenate((y_pred_test_1hr_prob[:,1], y_pred_gen_prob[:,1]), axis=0)

        auc_test_1hr = roc_auc_score(y_mix, y_mix_pred)

        df_RF_scores = pd.DataFrame(data=[[iter_id, fold, accuracy_score_train_st, accuracy_score_test_st, accuracy_score_gen, auc_train_st, auc_test_st, auc_test_1hr]], 
            columns=['iter_id', 'fold', 'accuracy_score_train_st', 'accuracy_score_test_st', 'accuracy_score_gen', 'auc_train_st', 'auc_test_st', 'auc_test_1hr'])
        
        save_path = os.path.join(current_file_path,'../EnGen_train_iterations/engen_output/Classification_results/')
        if os.path.exists(save_path+'/RF_scores_{}_clusters.csv'.format(n_clusters)):
            df_RF_scores.to_csv(save_path+'/RF_scores_{}_clusters.csv'.format(n_clusters), mode='a', index=False, header=False)
        else:
            os.makedirs(save_path, exist_ok=True)
            df_RF_scores.to_csv(save_path+'/RF_scores_{}_clusters.csv'.format(n_clusters), index=False, header=True)
        


if __name__ == "__main__":
    
    current_file_path = os.path.abspath(os.path.dirname(__file__))
    n_iterations = 30

    for iter_id in range(n_iterations):
        random_forrest_classifier_train(current_file_path, iter_id)