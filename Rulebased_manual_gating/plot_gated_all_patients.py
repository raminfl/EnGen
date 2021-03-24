import sys
import os
import glob
import pandas as pd 
import numpy as np 
import random
import re 
import seaborn as sns 
from matplotlib import pyplot as plt
from matplotlib import cm, ticker
import itertools
import pickle
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon, Point
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

def inv_arcsinh_transformation(x):
    a = 0
    b = 1/5
    c = 0
    
    return (np.sinh(x-c)-a)/b

def arcsinh_transformation(x):
    a = 0
    b = 1/5
    c = 0
    return np.arcsinh(a+b*x)+c


def get_feat_one_file(current_file_path, source_patient_id, df_sample, data_type, gates):

    save_sample_path = os.path.join(current_file_path,'../data/gates/{}/patient_{}/'.format(data_type, source_patient_id))
    os.makedirs(save_sample_path, exist_ok=True)
    df_sample.to_csv(save_sample_path+'Mononuclear.csv',index=False, header=True)
    eps = 0.0001 # handle zero in log scale
    feat_row = {}
    feat_row['patient_id'] = source_patient_id
    feat_row['data_type'] = data_type
    for cell_type in gates:
        if cell_type['parent'] == None:
            df = df_sample.copy()
        else:
            df = pd.read_csv(save_sample_path+'cells_type_{}.csv'.format(cell_type['parent']))
        

        df.reset_index(drop=True, inplace=True)
        df[df<eps] = eps
        # print(df)
        # print(df.columns)
        
        if cell_type['type'] == 'mononuclear_cells':
           
            points1 = [
                        (0+2*eps/3,1000),
                        (500,1000),
                        (500,40),
                        (10,20),
                        (0+2*eps/3,20)
                        ]
            polygon1 = Polygon(points1)

            # ax.set_title('Mononuclear Cells (CD45+CD66-)')
            inside = []
            for idx, row in df.iterrows():
                    
                inside.append(polygon1.contains(Point(row['139La_CD66'],row['115In_CD45'])))
            df_all = df.copy()
            df_all['inside']= inside
            # print(df1)  
            df1 = df_all[df_all['inside']==True]
            df_junk = df_all[df_all['inside']==False]
            df1.reset_index(drop=True, inplace=True)
            df_junk.reset_index(drop=True, inplace=True)
            df1.drop('inside', axis=1, inplace=True)
            # print('Mono freq = {}'.format(df1.shape[0]))
            # print('Junk freq = {}'.format(df_junk.shape[0]))
            feat_row['freq.mononuclear_cells'] = df1.shape[0]
            df1.to_csv(save_sample_path+'cells_type_{}.csv'.format(cell_type['type']), index=False, header=True)
            df1 = df1.applymap(arcsinh_transformation)
            for marker in df1.columns.values:
                feat_row['med.{}.mononuclear_cells'.format(marker)] = np.median(df1[marker])
            # print(feat_row)
            
        elif cell_type['type'] == 'Bcells_Tcells_CD19negCD3neg':
            
            points1 = [
                        (7000,0+2*eps/3),
                        (7000,200),
                        (100,200),
                        (100,0+2*eps/3)
                        ]
            polygon1 = Polygon(points1)

            points2 = [
                        (70,7),
                        (70,0+2*eps/3),
                        (0+2*eps/3,0+2*eps/3),
                        (0+2*eps/3,7)
                        ]
            polygon2 = Polygon(points2)

            points3 = [
                        (70,7),
                        (70,200),
                        (0+2*eps/3,200),
                        (0+2*eps/3,7)
                        ]
            polygon3 = Polygon(points3)

            inside1, inside2, inside3  = [], [], []

            for idx, row in df.iterrows():
                    
                inside1.append(polygon1.contains(Point(row['170Er_CD3'],row['142Nd_CD19'])))
                inside2.append(polygon2.contains(Point(row['170Er_CD3'],row['142Nd_CD19'])))
                inside3.append(polygon3.contains(Point(row['170Er_CD3'],row['142Nd_CD19'])))
            df_all = df.copy()
            df_all['inside1']= inside1
            df_all['inside2']= inside2
            df_all['inside3']= inside3
            # print(df1)  
            df1 = df_all[df_all['inside1']==True]
            df2 = df_all[df_all['inside2']==True]
            df3 = df_all[df_all['inside3']==True]
            df_junk = df_all[(df_all['inside1']==False) & (df_all['inside2']==False) & (df_all['inside3']==False)]
            df1.reset_index(drop=True, inplace=True)
            df2.reset_index(drop=True, inplace=True)
            df3.reset_index(drop=True, inplace=True)
            df_junk.reset_index(drop=True, inplace=True)
            df1.drop(['inside1', 'inside2', 'inside3'], axis=1, inplace=True)
            df2.drop(['inside1', 'inside2', 'inside3'], axis=1, inplace=True)
            df3.drop(['inside1', 'inside2', 'inside3'], axis=1, inplace=True)
            # print('Mono Tcells = {}'.format(df1.shape[0]))
            # print('Mono CD19negCD3neg = {}'.format(df2.shape[0]))
            # print('Mono Bcells = {}'.format(df3.shape[0]))
            # print('Junk freq = {}'.format(df_junk.shape[0]))


            feat_row['freq.Tcells'] = df1.shape[0]
            df1.to_csv(save_sample_path+'cells_type_{}__Tcells.csv'.format(cell_type['type']), index=False, header=True)
            df1 = df1.applymap(arcsinh_transformation)
            for marker in df1.columns.values:
                feat_row['med.{}.Tcells'.format(marker)] = np.median(df1[marker])

            feat_row['freq.CD19negCD3neg'] = df2.shape[0]
            df2.to_csv(save_sample_path+'cells_type_{}__CD19negCD3neg.csv'.format(cell_type['type']), index=False, header=True)
            df2 = df2.applymap(arcsinh_transformation)
            for marker in df2.columns.values:
                feat_row['med.{}.CD19negCD3neg'.format(marker)] = np.median(df2[marker])

            feat_row['freq.Bcells'] = df3.shape[0]
            df3.to_csv(save_sample_path+'cells_type_{}__Bcells.csv'.format(cell_type['type']), index=False, header=True)
            df3 = df3.applymap(arcsinh_transformation)
            for marker in df3.columns.values:
                feat_row['med.{}.Bcells'.format(marker)] = np.median(df3[marker])
            
            # print(feat_row)

        elif cell_type['type'] == 'Bcells':

            points1 = [
                        (0+2*eps/3,1),
                        (0+2*eps/3,100),
                        (2,100),
                        (2,1)
                        ]
            polygon1 = Polygon(points1)

            points2 = [
                        (2,1),
                        (2,100),
                        (1000,100),
                        (1000,1)
                        ]
            polygon2 = Polygon(points2)

            inside1, inside2  = [], []

            for idx, row in df.iterrows():
                    
                inside1.append(polygon1.contains(Point(row['171Yb_CD27'],row['142Nd_CD19'])))
                inside2.append(polygon2.contains(Point(row['171Yb_CD27'],row['142Nd_CD19'])))
            df_all = df.copy()
            df_all['inside1']= inside1
            df_all['inside2']= inside2
            # print(df1)  
            df1 = df_all[df_all['inside1']==True]
            df2 = df_all[df_all['inside2']==True]
            df_junk = df_all[(df_all['inside1']==False) & (df_all['inside2']==False)]
            df1.reset_index(drop=True, inplace=True)
            df2.reset_index(drop=True, inplace=True)
            df_junk.reset_index(drop=True, inplace=True)
            df1.drop(['inside1', 'inside2'], axis=1, inplace=True)
            df2.drop(['inside1', 'inside2'], axis=1, inplace=True)
            # print('Bcells Naive CD27- = {}'.format(df1.shape[0]))
            # print('Bcells Mem CD27+ = {}'.format(df2.shape[0]))
            # print('Junk freq = {}'.format(df_junk.shape[0]))

            feat_row['freq.Bcells_Naive'] = df1.shape[0]
            df1.to_csv(save_sample_path+'cells_type_{}__Naive.csv'.format(cell_type['type']), index=False, header=True)
            df1 = df1.applymap(arcsinh_transformation)
            for marker in df1.columns.values:
                feat_row['med.{}.Bcells_Naive'.format(marker)] = np.median(df1[marker])

            feat_row['freq.Bcells_Mem'] = df2.shape[0]
            df2.to_csv(save_sample_path+'cells_type_{}__Mem.csv'.format(cell_type['type']), index=False, header=True)
            df2 = df2.applymap(arcsinh_transformation)
            for marker in df2.columns.values:
                feat_row['med.{}.Bcells_Mem'.format(marker)] = np.median(df2[marker])
            

        elif cell_type['type'] == 'Tcells':
            
            points1 = [
                        (0+2*eps/3,0+2*eps/3),
                        (0+2*eps/3,30),
                        (30,30),
                        (30,0+2*eps/3)
                        ]
            polygon1 = Polygon(points1)

            points2 = [
                        (0+2*eps/3,30),
                        (0+2*eps/3,4000),
                        (30,4000),
                        (30,30)
                        ]
            polygon2 = Polygon(points2)
        
            points3 = [
                        (30,0+2*eps/3),
                        (4000,0+2*eps/3),
                        (4000,30),
                        (30,30)
                        ]
            polygon3 = Polygon(points3)
            

            inside1, inside2, inside3  = [], [], []

            for idx, row in df.iterrows():
                    
                inside1.append(polygon1.contains(Point(row['146Nd_CD8a'],row['145Nd_CD4'])))
                inside2.append(polygon2.contains(Point(row['146Nd_CD8a'],row['145Nd_CD4'])))
                inside3.append(polygon3.contains(Point(row['146Nd_CD8a'],row['145Nd_CD4'])))
            df_all = df.copy()
            df_all['inside1']= inside1
            df_all['inside2']= inside2
            df_all['inside3']= inside3
            # print(df1)  
            df1 = df_all[df_all['inside1']==True]
            df2 = df_all[df_all['inside2']==True]
            df3 = df_all[df_all['inside3']==True]
            df_junk = df_all[(df_all['inside1']==False) & (df_all['inside2']==False) & (df_all['inside3']==False)]
            df1.reset_index(drop=True, inplace=True)
            df2.reset_index(drop=True, inplace=True)
            df3.reset_index(drop=True, inplace=True)
            df_junk.reset_index(drop=True, inplace=True)
            df1.drop(['inside1', 'inside2', 'inside3'], axis=1, inplace=True)
            df2.drop(['inside1', 'inside2', 'inside3'], axis=1, inplace=True)
            df3.drop(['inside1', 'inside2', 'inside3'], axis=1, inplace=True)
            # print('CD4-CD8- = {}'.format(df1.shape[0]))
            # print('CD4+ Tcells = {}'.format(df2.shape[0]))
            # print('CD8+ Tcells = {}'.format(df3.shape[0]))
            # print('Junk freq = {}'.format(df_junk.shape[0]))
 

            feat_row['freq.Tcells_CD4negCD8neg'] = df1.shape[0]
            df1.to_csv(save_sample_path+'cells_type_{}__CD4negCD8neg.csv'.format(cell_type['type']), index=False, header=True)
            df1 = df1.applymap(arcsinh_transformation)
            for marker in df1.columns.values:
                feat_row['med.{}.Tcells_CD4negCD8neg'.format(marker)] = np.median(df1[marker])

            feat_row['freq.Tcells_CD4pos'] = df2.shape[0]
            df2.to_csv(save_sample_path+'cells_type_{}__CD4pos.csv'.format(cell_type['type']), index=False, header=True)
            df2 = df2.applymap(arcsinh_transformation)
            for marker in df2.columns.values:
                feat_row['med.{}.Tcells_CD4pos'.format(marker)] = np.median(df2[marker])

            feat_row['freq.Tcells_CD8pos'] = df3.shape[0]
            df3.to_csv(save_sample_path+'cells_type_{}__CD8pos.csv'.format(cell_type['type']), index=False, header=True)
            df3 = df3.applymap(arcsinh_transformation)
            for marker in df3.columns.values:
                feat_row['med.{}.Tcells_CD8pos'.format(marker)] = np.median(df3[marker])


        elif cell_type['type'] == 'TCRgd':
            
            points1 = [
                        (0.1,50),
                        (8000,50),
                        (8000,8000),
                        (0.1,8000)
                        ]
            polygon1 = Polygon(points1)
        
            inside1 = []

            for idx, row in df.iterrows():
                    
                inside1.append(polygon1.contains(Point(row['152Sm_TCRgd'],row['170Er_CD3'])))

            df_all = df.copy()
            df_all['inside1']= inside1

            # print(df1)  
            df1 = df_all[df_all['inside1']==True]
            df_junk = df_all[df_all['inside1']==False]
            df1.reset_index(drop=True, inplace=True)
            df_junk.reset_index(drop=True, inplace=True)
            df1.drop(['inside1'], axis=1, inplace=True)

            # print('TCRgd Tcells = {}'.format(df1.shape[0]))
            # print('Junk freq = {}'.format(df_junk.shape[0]))

            feat_row['freq.TCRgd'] = df1.shape[0]
            df1.to_csv(save_sample_path+'cells_type_{}.csv'.format(cell_type['type']), index=False, header=True)
            df1 = df1.applymap(arcsinh_transformation)
            for marker in df1.columns.values:
                feat_row['med.{}.TCRgd'.format(marker)] = np.median(df1[marker])


        elif cell_type['type'] == 'CD8pos':
    
            points1 = [
                        (0.5,0+2*eps/3),
                        (5000,0+2*eps/3),
                        (5000,10),
                        (0.5,10)
                        ]
            polygon1 = Polygon(points1)

            points2 = [
                        (0.5,10),
                        (5000,10),
                        (5000,1000),
                        (0.5,1000)
                        ]
            polygon2 = Polygon(points2)

            inside1, inside2  = [], []

            for idx, row in df.iterrows():
                    
                inside1.append(polygon1.contains(Point(row['146Nd_CD8a'],row['143Nd_CD45RA'])))
                inside2.append(polygon2.contains(Point(row['146Nd_CD8a'],row['143Nd_CD45RA'])))
            
            df_all = df.copy()
            df_all['inside1']= inside1
            df_all['inside2']= inside2
        
            # print(df1)  
            df1 = df_all[df_all['inside1']==True]
            df2 = df_all[df_all['inside2']==True]
        
            df_junk = df_all[(df_all['inside1']==False) & (df_all['inside2']==False)]
            df1.reset_index(drop=True, inplace=True)
            df2.reset_index(drop=True, inplace=True)
            df_junk.reset_index(drop=True, inplace=True)
            df1.drop(['inside1', 'inside2'], axis=1, inplace=True)
            df2.drop(['inside1', 'inside2'], axis=1, inplace=True)
            # print('CD8+ Mem Tcells = {}'.format(df1.shape[0]))
            # print('CD8+ Naive Tcells = {}'.format(df2.shape[0]))
            # print('Junk freq = {}'.format(df_junk.shape[0]))


            feat_row['freq.CD8posMem'] = df1.shape[0]
            df1.to_csv(save_sample_path+'cells_type_{}__CD8posMem.csv'.format(cell_type['type']), index=False, header=True)
            df1 = df1.applymap(arcsinh_transformation)
            for marker in df1.columns.values:
                feat_row['med.{}.CD8posMem'.format(marker)] = np.median(df1[marker])

            feat_row['freq.CD8posNaive'] = df2.shape[0]
            df2.to_csv(save_sample_path+'cells_type_{}__CD8posNaive.csv'.format(cell_type['type']), index=False, header=True)
            df2 = df2.applymap(arcsinh_transformation)
            for marker in df2.columns.values:
                feat_row['med.{}.CD8posNaive'.format(marker)] = np.median(df2[marker])


        elif cell_type['type'] == 'CD4pos':

            points1 = [
                        (0.5,0+2*eps/3),
                        (5000,0+2*eps/3),
                        (5000,10),
                        (0.5,10)
                        ]
            polygon1 = Polygon(points1)

            points2 = [
                        (0.5,10),
                        (5000,10),
                        (5000,1000),
                        (0.5,1000)
                        ]
            polygon2 = Polygon(points2)
    

            inside1, inside2  = [], []

            for idx, row in df.iterrows():
                    
                inside1.append(polygon1.contains(Point(row['145Nd_CD4'],row['143Nd_CD45RA'])))
                inside2.append(polygon2.contains(Point(row['145Nd_CD4'],row['143Nd_CD45RA'])))
            
            df_all = df.copy()
            df_all['inside1']= inside1
            df_all['inside2']= inside2
        
            # print(df1)  
            df1 = df_all[df_all['inside1']==True]
            df2 = df_all[df_all['inside2']==True]
        
            df_junk = df_all[(df_all['inside1']==False) & (df_all['inside2']==False)]
            df1.reset_index(drop=True, inplace=True)
            df2.reset_index(drop=True, inplace=True)
            df_junk.reset_index(drop=True, inplace=True)
            df1.drop(['inside1', 'inside2'], axis=1, inplace=True)
            df2.drop(['inside1', 'inside2'], axis=1, inplace=True)
            # print('CD4+ Mem Tcells = {}'.format(df1.shape[0]))
            # print('CD4+ Naive Tcells = {}'.format(df2.shape[0]))
            # print('Junk freq = {}'.format(df_junk.shape[0]))

        
            feat_row['freq.CD4posMem'] = df1.shape[0]
            df1.to_csv(save_sample_path+'cells_type_{}__CD4posMem.csv'.format(cell_type['type']), index=False, header=True)
            df1 = df1.applymap(arcsinh_transformation)
            for marker in df1.columns.values:
                feat_row['med.{}.CD4posMem'.format(marker)] = np.median(df1[marker])

            feat_row['freq.CD4posNaive'] = df2.shape[0]
            df2.to_csv(save_sample_path+'cells_type_{}__CD4posNaive.csv'.format(cell_type['type']), index=False, header=True)
            df2 = df2.applymap(arcsinh_transformation)
            for marker in df2.columns.values:
                feat_row['med.{}.CD4posNaive'.format(marker)] = np.median(df2[marker])

        elif cell_type['type'] == 'NKcells':
            
            points1 = [
                        (0+2*eps/3,0+2*eps/3),
                        (5000,0+2*eps/3),
                        (5000,10),
                        (0+2*eps/3,10)
                        ]
            polygon1 = Polygon(points1)

            points2 = [
                        (0+2*eps/3,10),
                        (100,10),
                        (100,5000),
                        (0+2*eps/3,5000)
                        ]
            polygon2 = Polygon(points2)

            inside1, inside2  = [], []

            for idx, row in df.iterrows():
                    
                inside1.append(polygon1.contains(Point(row['175Lu_CD14'],row['141Pr_CD7'])))
                inside2.append(polygon2.contains(Point(row['175Lu_CD14'],row['141Pr_CD7'])))
            
            df_all = df.copy()
            df_all['inside1']= inside1
            df_all['inside2']= inside2
        
            # print(df1)  
            df1 = df_all[df_all['inside1']==True]
            df2 = df_all[df_all['inside2']==True]
        
            df_junk = df_all[(df_all['inside1']==False) & (df_all['inside2']==False)]
            df1.reset_index(drop=True, inplace=True)
            df2.reset_index(drop=True, inplace=True)
            df_junk.reset_index(drop=True, inplace=True)
            df1.drop(['inside1', 'inside2'], axis=1, inplace=True)
            df2.drop(['inside1', 'inside2'], axis=1, inplace=True)
            # print('MC cells = {}'.format(df1.shape[0]))
            # print('NK cells (CD7+) = {}'.format(df2.shape[0]))
            # print('Junk freq = {}'.format(df_junk.shape[0]))

            feat_row['freq.MCcells'] = df1.shape[0]
            df1.to_csv(save_sample_path+'cells_type_{}__MCcells.csv'.format(cell_type['type']), index=False, header=True)
            df1 = df1.applymap(arcsinh_transformation)
            for marker in df1.columns.values:
                feat_row['med.{}.MCcells'.format(marker)] = np.median(df1[marker])

            feat_row['freq.NKcellsCD7pos'] = df2.shape[0]
            df2.to_csv(save_sample_path+'cells_type_{}__NKcellsCD7pos.csv'.format(cell_type['type']), index=False, header=True)
            df2 = df2.applymap(arcsinh_transformation)
            for marker in df2.columns.values:
                feat_row['med.{}.NKcellsCD7pos'.format(marker)] = np.median(df2[marker])

    return feat_row


def get_freqs_meds(current_file_path, gates, sample_types):
    # create a feature file after gating gt and gen from each patient in each cohort

    all_patients = pickle.load(open(os.path.join(current_file_path,'../EnGen_train_iterations/training_data/all_patient_ids.pkl'), 'rb'))['all_patient_ids']
    column_labels = ['149Sm_CREB', '150Nd_STAT5', '151Eu_p38', '153Eu_STAT1', '154Sm_STAT3', '155Gd_S6', '159Tb_MAPKAPK2', '164Dy_IkB', '166Er_NFkB', '167Er_ERK', '168Er_pSTAT6', '113In_CD235ab_CD61', '115In_CD45', '143Nd_CD45RA', '139La_CD66',
                    '141Pr_CD7', '142Nd_CD19', '144Nd_CD11b', '145Nd_CD4', '146Nd_CD8a', '147Sm_CD11c', '148Nd_CD123', '156Gd_CD24', '157Gd_CD161', '158Gd_CD33', '165Ho_CD16', '169Tm_CD25', '170Er_CD3', '171Yb_CD27', '172Yb_CD15',
                    '173Yb_CCR2', '175Lu_CD14',	'176Lu_CD56', '160Gd_Tbet', '162Dy_FoxP3', '152Sm_TCRgd', '174Yb_HLADR']
    df_feats = pd.DataFrame()


    for sample_type in sample_types:
        print('***************************************** getting features from sample_type {}'.format(sample_type))

        if sample_type == 'gt_1hr_all':
            df = pd.read_csv(os.path.join(current_file_path,'../data/gates/gt_Pre_1hr_all_patients.csv'))
            df = df[df['timepoint']=='1hr']
        
        for source_patient_id in all_patients:
            print('*************************** getting features from patient {}'.format(source_patient_id))
            if sample_type == 'gt_1hr_all':
                df_p = df.copy()
                df_p = df_p[df_p['patient_id']==source_patient_id]
                df_p.drop(['timepoint', 'patient_id'], axis=1, inplace=True)
                df_p = df_p.sample(n=20000, random_state=42)
                df_p.reset_index(drop=True, inplace=True)

            elif sample_type == 'gen_test_all_models':
                df_p = pd.read_csv(os.path.join(current_file_path,'../data/gates/gen_test/gen_test_{}_all_models_inv_scaled_inv_arcsinh.csv'.format(source_patient_id)))
                df_p = df_p.sample(n=20000, random_state=42)
                df_p.reset_index(drop=True, inplace=True)

            # df_p[df_p<eps] = eps
            feat_row = get_feat_one_file(current_file_path, source_patient_id, df_p, sample_type, gates)
            # print(feat_row)
            df_feats = df_feats.append(feat_row, ignore_index=True)
    os.makedirs(os.path.join(current_file_path,'../data/gates/comparison_plots/'), exist_ok=True)
    df_feats.to_csv(os.path.join(current_file_path,'../data/gates/comparison_plots/Features_med_freq_1hr_gt_vs_test_gen.csv'),index=False, header=True)
        

def plot_gates(current_file_path, data_type):
    #plot gates for fig 3 comparison of gated raw cells
    eps = 0.0001 # handle zero in log scale
    
    random.seed(42)
    all_patients = pickle.load(open(os.path.join(current_file_path,'../EnGen_train_iterations/training_data/all_patient_ids.pkl'), 'rb'))['all_patient_ids']
    column_labels = ['149Sm_CREB', '150Nd_STAT5', '151Eu_p38', '153Eu_STAT1', '154Sm_STAT3', '155Gd_S6', '159Tb_MAPKAPK2', '164Dy_IkB', '166Er_NFkB', '167Er_ERK', '168Er_pSTAT6', '113In_CD235ab_CD61', '115In_CD45', '143Nd_CD45RA', '139La_CD66',
                    '141Pr_CD7', '142Nd_CD19', '144Nd_CD11b', '145Nd_CD4', '146Nd_CD8a', '147Sm_CD11c', '148Nd_CD123', '156Gd_CD24', '157Gd_CD161', '158Gd_CD33', '165Ho_CD16', '169Tm_CD25', '170Er_CD3', '171Yb_CD27', '172Yb_CD15',
                    '173Yb_CCR2', '175Lu_CD14',	'176Lu_CD56', '160Gd_Tbet', '162Dy_FoxP3', '152Sm_TCRgd', '174Yb_HLADR']

    
    plot_patient = random.sample(all_patients, 1)[0] # select one patient randomly for plotting
    print('plot_patient = {}'.format(plot_patient))
    
    # for patient_id in [plot_patient]: # only plot the randomly selected patient
    for patient_id in all_patients: 
    
        # plotting Mononuclear
        df = pd.read_csv(os.path.join(current_file_path,'../data/gates/{}/patient_{}/Mononuclear.csv'.format(data_type, patient_id)))
        df.reset_index(drop=True, inplace=True)
        df[df<eps] = eps

        fig, ax = plt.subplots(figsize=(4,4))

        df.plot(kind='scatter', x='139La_CD66', y='115In_CD45', ax=ax, c='black', s=1)
        points1 = [
                    (0+2*eps/3,1000),
                    (500,1000),
                    (500,40),
                    (10,20),
                    (0+2*eps/3,20)
                    ]
        polygon1 = Polygon(points1)

        x,y = polygon1.exterior.xy
        ax.plot(x,y, c='#00b894')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim((eps/2, 10*1000))
        ax.set_ylim((eps/2, 10*1000))
        # ax.set_xticks([20, 200, 500])
        # ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.set_title('Mononuclear Cells (CD45+CD66-)')
        inside = []
        # plt.scatter(x=100, y=100, c='red', s=15)
        
        for idx, row in df.iterrows():
            # print(idx)
            # print(row)
            # ax.scatter(x=row['139La_CD66'], y=row['115In_CD45'], c='black', s=5)
            if not polygon1.contains(Point(row['139La_CD66'],row['115In_CD45'])):
                print(row['139La_CD66'], row['115In_CD45'])
                
            inside.append(polygon1.contains(Point(row['139La_CD66'],row['115In_CD45'])))
        df_all = df.copy()
        df_all['inside']= inside
        # print(df1)  
        df1 = df_all[df_all['inside']==True]
        df_junk = df_all[df_all['inside']==False]
        df1.reset_index(drop=True, inplace=True)
        df_junk.reset_index(drop=True, inplace=True)
        df1.drop('inside', axis=1, inplace=True)
        print('Mono freq = {}'.format(df1.shape[0]))
        print('Junk freq = {}'.format(df_junk.shape[0]))
        
        plt.savefig(os.path.join(current_file_path,'../data/gates/{}/patient_{}/cell_type_Mononuclear.pdf'.format(data_type, patient_id)), format='pdf', dpi=600)
        plt.close()

        # ---------------------------------------------------------------------------------------------------
        # ploting 'Bcells_Tcells_CD19negCD3neg'
        df =  pd.read_csv(os.path.join(current_file_path,'../data/gates/{}/patient_{}/cells_type_mononuclear_cells.csv'.format(data_type, patient_id)))
        df.reset_index(drop=True, inplace=True)
        df[df<eps] = eps

        fig, ax = plt.subplots(figsize=(4,4))
        df.plot(kind='scatter', x='170Er_CD3', y='142Nd_CD19', ax=ax, c='black', s=1)
        points1 = [
                    (7000,0+2*eps/3),
                    (7000,200),
                    (100,200),
                    (100,0+2*eps/3)
                    ]
        polygon1 = Polygon(points1)
        x1,y1 = polygon1.exterior.xy
        ax.plot(x1,y1, c='#00b894')

        points2 = [
                    (70,7),
                    (70,0+2*eps/3),
                    (0+2*eps/3,0+2*eps/3),
                    (0+2*eps/3,7)
                    ]
        polygon2 = Polygon(points2)
        x2,y2 = polygon2.exterior.xy
        ax.plot(x2,y2, c='#00b894')

        points3 = [
                    (70,7),
                    (70,200),
                    (0+2*eps/3,200),
                    (0+2*eps/3,7)
                    ]
        polygon3 = Polygon(points3)
        x3,y3 = polygon3.exterior.xy
        ax.plot(x3,y3, c='#00b894')

        

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim((eps/2, 10*1000))
        ax.set_ylim((eps/2, 10*1000))
        ax.set_title('B cells (CD19+CD3-)\nTcells (CD19-CD3+)\nCD19-CD3-')
        inside1, inside2, inside3  = [], [], []

        for idx, row in df.iterrows():
                
            inside1.append(polygon1.contains(Point(row['170Er_CD3'],row['142Nd_CD19'])))
            inside2.append(polygon2.contains(Point(row['170Er_CD3'],row['142Nd_CD19'])))
            inside3.append(polygon3.contains(Point(row['170Er_CD3'],row['142Nd_CD19'])))
        df_all = df.copy()
        df_all['inside1']= inside1
        df_all['inside2']= inside2
        df_all['inside3']= inside3
        # print(df1)  
        df1 = df_all[df_all['inside1']==True]
        df2 = df_all[df_all['inside2']==True]
        df3 = df_all[df_all['inside3']==True]
        df_junk = df_all[(df_all['inside1']==False) & (df_all['inside2']==False) & (df_all['inside3']==False)]
        df1.reset_index(drop=True, inplace=True)
        df2.reset_index(drop=True, inplace=True)
        df3.reset_index(drop=True, inplace=True)
        df_junk.reset_index(drop=True, inplace=True)
        df1.drop(['inside1', 'inside2', 'inside3'], axis=1, inplace=True)
        df2.drop(['inside1', 'inside2', 'inside3'], axis=1, inplace=True)
        df3.drop(['inside1', 'inside2', 'inside3'], axis=1, inplace=True)
        print('Mono Tcells = {}'.format(df1.shape[0]))
        print('Mono CD19negCD3neg = {}'.format(df2.shape[0]))
        print('Mono Bcells = {}'.format(df3.shape[0]))
        print('Junk freq = {}'.format(df_junk.shape[0]))

        # style = dict(size=2, color='#00b894')
        # ax.text(1000, 300, 'Tcells ({})'.format(df1.shape[0]), ha='center', **style)
        # ax.text(0.01, 0.001, 'CD19negCD3neg ({})'.format(df2.shape[0]), ha='center', **style)
        # ax.text(10, 300, 'Bcells ({})'.format(df3.shape[0]), ha='center', **style)

        # df1.to_csv('/home/raminf/DL/FCS_files_kehlet/iter_33pc_train/plots/gates/{}/cells_type_{}__Tcells.csv'.format(data_type,cell_type['type']), index=False, header=True)
        # df2.to_csv('/home/raminf/DL/FCS_files_kehlet/iter_33pc_train/plots/gates/{}/cells_type_{}__CD19negCD3neg.csv'.format(data_type,cell_type['type']), index=False, header=True)
        # df3.to_csv('/home/raminf/DL/FCS_files_kehlet/iter_33pc_train/plots/gates/{}/cells_type_{}__Bcells.csv'.format(data_type,cell_type['type']), index=False, header=True)
        plt.savefig(os.path.join(current_file_path,'../data/gates/{}/patient_{}/cell_type_Bcells_Tcells_CD19negCD3neg.pdf'.format(data_type, patient_id)), format='pdf', dpi=600)
        plt.close()

    # ---------------------------------------------------------------------------------------------------
        # ploting 'Bcells':
        df =  pd.read_csv(os.path.join(current_file_path,'../data/gates/{}/patient_{}/cells_type_Bcells_Tcells_CD19negCD3neg__Bcells.csv'.format(data_type, patient_id)))
        df.reset_index(drop=True, inplace=True)
        df[df<eps] = eps

        fig, ax = plt.subplots(figsize=(4,4))
        df.plot(kind='scatter', x='171Yb_CD27', y='142Nd_CD19', ax=ax, c='black', s=1)
        points1 = [
                    (0+2*eps/3,1),
                    (0+2*eps/3,100),
                    (2,100),
                    (2,1)
                    ]
        polygon1 = Polygon(points1)
        x1,y1 = polygon1.exterior.xy
        ax.plot(x1,y1, c='#00b894')

        points2 = [
                    (2,1),
                    (2,100),
                    (1000,100),
                    (1000,1)
                    ]
        polygon2 = Polygon(points2)
        x2,y2 = polygon2.exterior.xy
        ax.plot(x2,y2, c='#00b894')

        

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim((eps/2, 10*1000))
        ax.set_ylim((eps/2, 10*1000))
        ax.set_title('Bcells Mem CD27+\nBcells Naive CD27-')
        inside1, inside2  = [], []

        for idx, row in df.iterrows():
                
            inside1.append(polygon1.contains(Point(row['171Yb_CD27'],row['142Nd_CD19'])))
            inside2.append(polygon2.contains(Point(row['171Yb_CD27'],row['142Nd_CD19'])))
        df_all = df.copy()
        df_all['inside1']= inside1
        df_all['inside2']= inside2
        # print(df1)  
        df1 = df_all[df_all['inside1']==True]
        df2 = df_all[df_all['inside2']==True]
        df_junk = df_all[(df_all['inside1']==False) & (df_all['inside2']==False)]
        df1.reset_index(drop=True, inplace=True)
        df2.reset_index(drop=True, inplace=True)
        df_junk.reset_index(drop=True, inplace=True)
        df1.drop(['inside1', 'inside2'], axis=1, inplace=True)
        df2.drop(['inside1', 'inside2'], axis=1, inplace=True)
        print('Bcells Naive CD27- = {}'.format(df1.shape[0]))
        print('Bcells Mem CD27+ = {}'.format(df2.shape[0]))
        print('Junk freq = {}'.format(df_junk.shape[0]))

        plt.savefig(os.path.join(current_file_path,'../data/gates/{}/patient_{}/cell_type_Bcells.pdf'.format(data_type, patient_id)), format='pdf', dpi=600)
        plt.close()

        # ---------------------------------------------------------------------------------------------------
        # ploting 'Tcells':
        df =  pd.read_csv(os.path.join(current_file_path,'../data/gates/{}/patient_{}/cells_type_Bcells_Tcells_CD19negCD3neg__Tcells.csv'.format(data_type, patient_id)))
        df.reset_index(drop=True, inplace=True)
        df[df<eps] = eps

        fig, ax = plt.subplots(figsize=(4,4))
        df.plot(kind='scatter', x='146Nd_CD8a', y='145Nd_CD4', ax=ax, c='black', s=1)
        points1 = [
                    (0+2*eps/3,0+2*eps/3),
                    (0+2*eps/3,30),
                    (30,30),
                    (30,0+2*eps/3)
                    ]
        polygon1 = Polygon(points1)
        x1,y1 = polygon1.exterior.xy
        ax.plot(x1,y1, c='#00b894')

        points2 = [
                    (0+2*eps/3,30),
                    (0+2*eps/3,4000),
                    (30,4000),
                    (30,30)
                    ]
        polygon2 = Polygon(points2)
        x2,y2 = polygon2.exterior.xy
        ax.plot(x2,y2, c='#00b894')

        points3 = [
                    (30,0+2*eps/3),
                    (4000,0+2*eps/3),
                    (4000,30),
                    (30,30)
                    ]
        polygon3 = Polygon(points3)
        x3,y3 = polygon3.exterior.xy
        ax.plot(x3,y3, c='#00b894')

        

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim((eps/2, 10*1000))
        ax.set_ylim((eps/2, 10*1000))
        ax.set_title('CD4+ Tcells\nCD8+ Tcells\nCD4-CD8-')
        inside1, inside2, inside3  = [], [], []

        for idx, row in df.iterrows():
                
            inside1.append(polygon1.contains(Point(row['146Nd_CD8a'],row['145Nd_CD4'])))
            inside2.append(polygon2.contains(Point(row['146Nd_CD8a'],row['145Nd_CD4'])))
            inside3.append(polygon3.contains(Point(row['146Nd_CD8a'],row['145Nd_CD4'])))
        df_all = df.copy()
        df_all['inside1']= inside1
        df_all['inside2']= inside2
        df_all['inside3']= inside3
        # print(df1)  
        df1 = df_all[df_all['inside1']==True]
        df2 = df_all[df_all['inside2']==True]
        df3 = df_all[df_all['inside3']==True]
        df_junk = df_all[(df_all['inside1']==False) & (df_all['inside2']==False) & (df_all['inside3']==False)]
        df1.reset_index(drop=True, inplace=True)
        df2.reset_index(drop=True, inplace=True)
        df3.reset_index(drop=True, inplace=True)
        df_junk.reset_index(drop=True, inplace=True)
        df1.drop(['inside1', 'inside2', 'inside3'], axis=1, inplace=True)
        df2.drop(['inside1', 'inside2', 'inside3'], axis=1, inplace=True)
        df3.drop(['inside1', 'inside2', 'inside3'], axis=1, inplace=True)
        print('CD4-CD8- = {}'.format(df1.shape[0]))
        print('CD4+ Tcells = {}'.format(df2.shape[0]))
        print('CD8+ Tcells = {}'.format(df3.shape[0]))
        print('Junk freq = {}'.format(df_junk.shape[0]))

        plt.savefig(os.path.join(current_file_path,'../data/gates/{}/patient_{}/cell_type_Tcells.pdf'.format(data_type, patient_id)), format='pdf', dpi=600)
        plt.close()

        # ---------------------------------------------------------------------------------------------------
        # ploting 'TCRgd':
        df =  pd.read_csv(os.path.join(current_file_path,'../data/gates/{}/patient_{}/cells_type_Tcells__CD4negCD8neg.csv'.format(data_type, patient_id)))
        df.reset_index(drop=True, inplace=True)
        df[df<eps] = eps

        fig, ax = plt.subplots(figsize=(4,4))
        df.plot(kind='scatter', x='152Sm_TCRgd', y='170Er_CD3', ax=ax, c='black', s=1)
        points1 = [
                    (0.1,50),
                    (8000,50),
                    (8000,8000),
                    (0.1,8000)
                    ]
        polygon1 = Polygon(points1)
        x1,y1 = polygon1.exterior.xy
        ax.plot(x1,y1, c='#00b894')

        

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim((eps/2, 10*1000))
        ax.set_ylim((eps/2, 10*1000))
        ax.set_title('TCRgd Tcells')
        inside1 = []

        for idx, row in df.iterrows():
                
            inside1.append(polygon1.contains(Point(row['152Sm_TCRgd'],row['170Er_CD3'])))

        df_all = df.copy()
        df_all['inside1']= inside1

        # print(df1)  
        df1 = df_all[df_all['inside1']==True]
        df_junk = df_all[df_all['inside1']==False]
        df1.reset_index(drop=True, inplace=True)
        df_junk.reset_index(drop=True, inplace=True)
        df1.drop(['inside1'], axis=1, inplace=True)

        print('TCRgd Tcells = {}'.format(df1.shape[0]))
        print('Junk freq = {}'.format(df_junk.shape[0]))

        plt.savefig(os.path.join(current_file_path,'../data/gates/{}/patient_{}/cell_type_TCRgd.pdf'.format(data_type, patient_id)), format='pdf', dpi=600)
        plt.close()

        # ---------------------------------------------------------------------------------------------------
        # ploting 'CD8pos':
        df =  pd.read_csv(os.path.join(current_file_path,'../data/gates/{}/patient_{}/cells_type_Tcells__CD8pos.csv'.format(data_type, patient_id)))
        df.reset_index(drop=True, inplace=True)
        df[df<eps] = eps

        fig, ax = plt.subplots(figsize=(4,4))
        df.plot(kind='scatter', x='146Nd_CD8a', y='143Nd_CD45RA', ax=ax, c='black', s=1)
        points1 = [
                    (0.5,0+2*eps/3),
                    (5000,0+2*eps/3),
                    (5000,10),
                    (0.5,10)
                    ]
        polygon1 = Polygon(points1)
        x1,y1 = polygon1.exterior.xy
        ax.plot(x1,y1, c='#00b894')

        points2 = [
                    (0.5,10),
                    (5000,10),
                    (5000,1000),
                    (0.5,1000)
                    ]
        polygon2 = Polygon(points2)
        x2,y2 = polygon2.exterior.xy
        ax.plot(x2,y2, c='#00b894')

      

        

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim((eps/2, 10*1000))
        ax.set_ylim((eps/2, 10*1000))
        ax.set_title('CD8+ Mem Tcells\nCD8+ Naive Tcells')
        inside1, inside2  = [], []

        for idx, row in df.iterrows():
            # print(idx)
            # print(row)
            # if not polygon1.contains(Point(row['170Er_CD3'],row['142Nd_CD19'])):
                # print(row['170Er_CD3'], row['142Nd_CD19'])
                
            inside1.append(polygon1.contains(Point(row['146Nd_CD8a'],row['143Nd_CD45RA'])))
            inside2.append(polygon2.contains(Point(row['146Nd_CD8a'],row['143Nd_CD45RA'])))
           
        df_all = df.copy()
        df_all['inside1']= inside1
        df_all['inside2']= inside2
      
        # print(df1)  
        df1 = df_all[df_all['inside1']==True]
        df2 = df_all[df_all['inside2']==True]
     
        df_junk = df_all[(df_all['inside1']==False) & (df_all['inside2']==False)]
        df1.reset_index(drop=True, inplace=True)
        df2.reset_index(drop=True, inplace=True)
        df_junk.reset_index(drop=True, inplace=True)
        df1.drop(['inside1', 'inside2'], axis=1, inplace=True)
        df2.drop(['inside1', 'inside2'], axis=1, inplace=True)
        print('CD8+ Mem Tcells = {}'.format(df1.shape[0]))
        print('CD8+ Naive Tcells = {}'.format(df2.shape[0]))
        print('Junk freq = {}'.format(df_junk.shape[0]))

        plt.savefig(os.path.join(current_file_path,'../data/gates/{}/patient_{}/cell_type_CD8pos.pdf'.format(data_type, patient_id)), format='pdf', dpi=600)
        plt.close()

        # ---------------------------------------------------------------------------------------------------
        # ploting 'CD4pos':
        df =  pd.read_csv(os.path.join(current_file_path,'../data/gates/{}/patient_{}/cells_type_Tcells__CD4pos.csv'.format(data_type, patient_id)))
        df.reset_index(drop=True, inplace=True)
        df[df<eps] = eps

        fig, ax = plt.subplots(figsize=(4,4))
        df.plot(kind='scatter', x='145Nd_CD4', y='143Nd_CD45RA', ax=ax, c='black', s=1)
        points1 = [
                    (0.5,0+2*eps/3),
                    (5000,0+2*eps/3),
                    (5000,10),
                    (0.5,10)
                    ]
        polygon1 = Polygon(points1)
        x1,y1 = polygon1.exterior.xy
        ax.plot(x1,y1, c='#00b894')

        points2 = [
                    (0.5,10),
                    (5000,10),
                    (5000,1000),
                    (0.5,1000)
                    ]
        polygon2 = Polygon(points2)
        x2,y2 = polygon2.exterior.xy
        ax.plot(x2,y2, c='#00b894')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim((eps/2, 10*1000))
        ax.set_ylim((eps/2, 10*1000))
        ax.set_title('CD4+ Mem Tcells\nCD4+ Naive Tcells')
        inside1, inside2  = [], []

        for idx, row in df.iterrows():
          
                
            inside1.append(polygon1.contains(Point(row['145Nd_CD4'],row['143Nd_CD45RA'])))
            inside2.append(polygon2.contains(Point(row['145Nd_CD4'],row['143Nd_CD45RA'])))
           
        df_all = df.copy()
        df_all['inside1']= inside1
        df_all['inside2']= inside2
      
        # print(df1)  
        df1 = df_all[df_all['inside1']==True]
        df2 = df_all[df_all['inside2']==True]
     
        df_junk = df_all[(df_all['inside1']==False) & (df_all['inside2']==False)]
        df1.reset_index(drop=True, inplace=True)
        df2.reset_index(drop=True, inplace=True)
        df_junk.reset_index(drop=True, inplace=True)
        df1.drop(['inside1', 'inside2'], axis=1, inplace=True)
        df2.drop(['inside1', 'inside2'], axis=1, inplace=True)
        print('CD4+ Mem Tcells = {}'.format(df1.shape[0]))
        print('CD4+ Naive Tcells = {}'.format(df2.shape[0]))
        print('Junk freq = {}'.format(df_junk.shape[0]))

        plt.savefig(os.path.join(current_file_path,'../data/gates/{}/patient_{}/cell_type_CD4pos.pdf'.format(data_type, patient_id)), format='pdf', dpi=600)
        plt.close()

        # ---------------------------------------------------------------------------------------------------
        # ploting 'NKcells':
        df =  pd.read_csv(os.path.join(current_file_path,'../data/gates/{}/patient_{}/cells_type_Bcells_Tcells_CD19negCD3neg__CD19negCD3neg.csv'.format(data_type, patient_id)))
        df.reset_index(drop=True, inplace=True)
        df[df<eps] = eps

        fig, ax = plt.subplots(figsize=(4,4)) 
        df.plot(kind='scatter', x='175Lu_CD14', y='141Pr_CD7', ax=ax, c='black', s=1)
        points1 = [
                    (0+2*eps/3,0+2*eps/3),
                    (5000,0+2*eps/3),
                    (5000,10),
                    (0+2*eps/3,10)
                    ]
        polygon1 = Polygon(points1)
        x1,y1 = polygon1.exterior.xy
        ax.plot(x1,y1, c='#00b894')

        points2 = [
                    (0+2*eps/3,10),
                    (100,10),
                    (100,5000),
                    (0+2*eps/3,5000)
                    ]
        polygon2 = Polygon(points2)
        x2,y2 = polygon2.exterior.xy
        ax.plot(x2,y2, c='#00b894')
        

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim((eps/2, 10*1000))
        ax.set_ylim((eps/2, 10*1000))
        ax.set_title('NK cells (CD7+)\nMC cells')
        inside1, inside2  = [], []

        for idx, row in df.iterrows():
                
            inside1.append(polygon1.contains(Point(row['175Lu_CD14'],row['141Pr_CD7'])))
            inside2.append(polygon2.contains(Point(row['175Lu_CD14'],row['141Pr_CD7'])))
           
        df_all = df.copy()
        df_all['inside1']= inside1
        df_all['inside2']= inside2
      
        # print(df1)  
        df1 = df_all[df_all['inside1']==True]
        df2 = df_all[df_all['inside2']==True]
     
        df_junk = df_all[(df_all['inside1']==False) & (df_all['inside2']==False)]
        df1.reset_index(drop=True, inplace=True)
        df2.reset_index(drop=True, inplace=True)
        df_junk.reset_index(drop=True, inplace=True)
        df1.drop(['inside1', 'inside2'], axis=1, inplace=True)
        df2.drop(['inside1', 'inside2'], axis=1, inplace=True)
        print('MC cells = {}'.format(df1.shape[0]))
        print('NK cells (CD7+) = {}'.format(df2.shape[0]))
        print('Junk freq = {}'.format(df_junk.shape[0]))

        plt.savefig(os.path.join(current_file_path,'../data/gates/{}/patient_{}/cell_type_NKcells.pdf'.format(data_type, patient_id)), format='pdf', dpi=600)
        plt.close()

    

        


if __name__ == "__main__":
    
    gates = []
    cell_type = {'type':'mononuclear_cells', 'parent':None}
    gates.append(cell_type)
    cell_type = {'type':'Bcells_Tcells_CD19negCD3neg', 'parent':'mononuclear_cells'}
    gates.append(cell_type)
    cell_type = {'type':'Bcells', 'parent':'Bcells_Tcells_CD19negCD3neg__Bcells'}
    gates.append(cell_type)
    cell_type = {'type':'Tcells', 'parent':'Bcells_Tcells_CD19negCD3neg__Tcells'}
    gates.append(cell_type)
    cell_type = {'type':'TCRgd', 'parent':'Tcells__CD4negCD8neg'}
    gates.append(cell_type)
    cell_type = {'type':'CD8pos', 'parent':'Tcells__CD8pos'}
    gates.append(cell_type)
    cell_type = {'type':'CD4pos', 'parent':'Tcells__CD4pos'}
    gates.append(cell_type)
    cell_type = {'type':'NKcells', 'parent':'Bcells_Tcells_CD19negCD3neg__CD19negCD3neg'}
    gates.append(cell_type)


    current_file_path = os.path.abspath(os.path.dirname(__file__))

    get_freqs_meds(current_file_path, gates, ['gt_1hr_all', 'gen_test_all_models'])

    # for st in ['gt_Pre_all', 'gt_1hr_all', 'gen_train_all_models', 'gen_test_all_models']:
        
    #     plot_gates(current_file_path, st)







    
    
