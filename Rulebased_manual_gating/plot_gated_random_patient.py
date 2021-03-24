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
    # if np.isinf((np.sinh(x-c)-a)/b):
    #     print(x)
    #     print((np.sinh(x-c)-a)/b)
        # sys.exit()
    return (np.sinh(x-c)-a)/b

def arcsinh_transformation(x):
    a = 0
    b = 1/5
    c = 0
    return np.arcsinh(a+b*x)+c


def plot_gates(current_file_path, data_type, cell_type):
    #plot gates 
    eps = 0.0001 # handle zero in log scale

    if cell_type['parent'] == None:
        if data_type == 'gt_Pre_all':
            df = pd.read_csv(os.path.join(current_file_path,'../data/gates/gt_Pre_1hr_all_patients.csv'))
            df = df[df['timepoint']=='Pre']
            df.drop(['timepoint', 'patient_id'], axis=1, inplace=True)
        elif data_type == 'gt_1hr_all':
            df = pd.read_csv(os.path.join(current_file_path,'../data/gates/gt_Pre_1hr_all_patients.csv'))
            df = df[df['timepoint']=='1hr']
            df.drop(['timepoint', 'patient_id'], axis=1, inplace=True)
        elif data_type == 'gen_train_all_models':
            all_patients = pickle.load(open(os.path.join(current_file_path,'../EnGen_train_iterations/training_data/all_patient_ids.pkl'), 'rb'))['all_patient_ids']
            patient_id = random.sample(all_patients, 1)[0] # pick on patient randomly for visualization
            df = pd.read_csv(os.path.join(current_file_path,'../data/gates/gen_train/gen_train_{}_all_models_inv_scaled_inv_arcsinh.csv'.format(patient_id)))
            # print(df)
        elif data_type == 'gen_test_all_models':
            all_patients = pickle.load(open(os.path.join(current_file_path,'../EnGen_train_iterations/training_data/all_patient_ids.pkl'), 'rb'))['all_patient_ids']
            patient_id = random.sample(all_patients, 1)[0] # pick on patient randomly for visualization
            df = pd.read_csv(os.path.join(current_file_path,'../data/gates/gen_test/gen_test_{}_all_models_inv_scaled_inv_arcsinh.csv'.format(patient_id)))
            # print(df)
        else:
            print('Data type not defined')
            sys.exit()
        df = df.sample(n=20000, random_state=42)
    else:
        df = pd.read_csv(os.path.join(current_file_path,'../data/gates/{}/cells_type_{}.csv'.format(data_type,cell_type['parent'])))
    

    df.reset_index(drop=True, inplace=True)
    df[df<eps] = eps
    
    fig, ax = plt.subplots(figsize=(7,7))


    if cell_type['type'] == 'mononuclear_cells':
        df.plot(kind='scatter', x='139La_CD66', y='115In_CD45', ax=ax, c='black', s=5)
        points1 = [
                    (0+2*eps/3,1000),
                    (500,1000),
                    (500,40),
                    (10,20),
                    (0+2*eps/3,20)
                    ]
        polygon1 = Polygon(points1)

        x,y = polygon1.exterior.xy
        ax.plot(x,y, c='#d62828')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim((eps/2, 10*1000))
        ax.set_ylim((eps/2, 10*1000))
        # ax.set_xticks([20, 200, 500])
        # ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.set_title('Mononuclear Cells (CD45+CD66-)')
        inside = []
        
        for idx, row in df.iterrows():
            
            if not polygon1.contains(Point(row['139La_CD66'],row['115In_CD45'])):
                print(row['139La_CD66'], row['115In_CD45'])
                
            inside.append(polygon1.contains(Point(row['139La_CD66'],row['115In_CD45'])))
        df_all = df.copy()
        df_all['inside']= inside
          
        df1 = df_all[df_all['inside']==True]
        df_junk = df_all[df_all['inside']==False]
        df1.reset_index(drop=True, inplace=True)
        df_junk.reset_index(drop=True, inplace=True)
        df1.drop('inside', axis=1, inplace=True)
        print('Mono freq = {}'.format(df1.shape[0]))
        print('Junk freq = {}'.format(df_junk.shape[0]))
        style = dict(size=10, color='#d62828')
        ax.text(100,2000, 'Mononuclear Cells ({})'.format(df1.shape[0]), ha='center', **style)
        os.makedirs(os.path.join(current_file_path,'../data/gates/{}/'.format(data_type)), exist_ok=True)
        df1.to_csv(os.path.join(current_file_path,'../data/gates/{}/cells_type_{}.csv'.format(data_type,cell_type['type'])), index=False, header=True)

    elif cell_type['type'] == 'Bcells_Tcells_CD19negCD3neg':
        df.plot(kind='scatter', x='170Er_CD3', y='142Nd_CD19', ax=ax, c='black', s=5)
        points1 = [
                    (7000,0+2*eps/3),
                    (7000,200),
                    (100,200),
                    (100,0+2*eps/3)
                    ]
        polygon1 = Polygon(points1)
        x1,y1 = polygon1.exterior.xy
        ax.plot(x1,y1, c='#d62828')

        points2 = [
                    (70,7),
                    (70,0+2*eps/3),
                    (0+2*eps/3,0+2*eps/3),
                    (0+2*eps/3,7)
                    ]
        polygon2 = Polygon(points2)
        x2,y2 = polygon2.exterior.xy
        ax.plot(x2,y2, c='#d62828')

        points3 = [
                    (70,7),
                    (70,200),
                    (0+2*eps/3,200),
                    (0+2*eps/3,7)
                    ]
        polygon3 = Polygon(points3)
        x3,y3 = polygon3.exterior.xy
        ax.plot(x3,y3, c='#d62828')

        

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

        style = dict(size=10, color='#d62828')
        ax.text(1000, 300, 'Tcells ({})'.format(df1.shape[0]), ha='center', **style)
        ax.text(0.01, 0.001, 'CD19negCD3neg ({})'.format(df2.shape[0]), ha='center', **style)
        ax.text(10, 300, 'Bcells ({})'.format(df3.shape[0]), ha='center', **style)

        df1.to_csv(os.path.join(current_file_path,'../data/gates/{}/cells_type_{}__Tcells.csv'.format(data_type,cell_type['type'])), index=False, header=True)
        df2.to_csv(os.path.join(current_file_path,'../data/gates/{}/cells_type_{}__CD19negCD3neg.csv'.format(data_type,cell_type['type'])), index=False, header=True)
        df3.to_csv(os.path.join(current_file_path,'../data/gates/{}/cells_type_{}__Bcells.csv'.format(data_type,cell_type['type'])), index=False, header=True)

    elif cell_type['type'] == 'Bcells':
        df.plot(kind='scatter', x='171Yb_CD27', y='142Nd_CD19', ax=ax, c='black', s=5)
        points1 = [
                    (0+2*eps/3,1),
                    (0+2*eps/3,100),
                    (2,100),
                    (2,1)
                    ]
        polygon1 = Polygon(points1)
        x1,y1 = polygon1.exterior.xy
        ax.plot(x1,y1, c='#d62828')

        points2 = [
                    (2,1),
                    (2,100),
                    (1000,100),
                    (1000,1)
                    ]
        polygon2 = Polygon(points2)
        x2,y2 = polygon2.exterior.xy
        ax.plot(x2,y2, c='#d62828')

        

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

        style = dict(size=10, color='#d62828')
        ax.text(0.001, 300, 'Bcells Naive CD27- ({})'.format(df1.shape[0]), ha='center', **style)
        ax.text(1000, 300, 'Bcells Mem CD27+ ({})'.format(df2.shape[0]), ha='center', **style)

        df1.to_csv(os.path.join(current_file_path,'../data/gates/{}/cells_type_{}__Naive.csv'.format(data_type,cell_type['type'])), index=False, header=True)
        df2.to_csv(os.path.join(current_file_path,'../data/gates/{}/cells_type_{}__Mem.csv'.format(data_type,cell_type['type'])), index=False, header=True)

    elif cell_type['type'] == 'Tcells':
        df.plot(kind='scatter', x='146Nd_CD8a', y='145Nd_CD4', ax=ax, c='black', s=5)
        points1 = [
                    (0+2*eps/3,0+2*eps/3),
                    (0+2*eps/3,30),
                    (30,30),
                    (30,0+2*eps/3)
                    ]
        polygon1 = Polygon(points1)
        x1,y1 = polygon1.exterior.xy
        ax.plot(x1,y1, c='#d62828')

        points2 = [
                    (0+2*eps/3,30),
                    (0+2*eps/3,4000),
                    (30,4000),
                    (30,30)
                    ]
        polygon2 = Polygon(points2)
        x2,y2 = polygon2.exterior.xy
        ax.plot(x2,y2, c='#d62828')

        points3 = [
                    (30,0+2*eps/3),
                    (4000,0+2*eps/3),
                    (4000,30),
                    (30,30)
                    ]
        polygon3 = Polygon(points3)
        x3,y3 = polygon3.exterior.xy
        ax.plot(x3,y3, c='#d62828')

        

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

        style = dict(size=10, color='#d62828')
        ax.text(0.01, 0.001, 'CD4-CD8- Tcells ({})'.format(df1.shape[0]), ha='center', **style)
        ax.text(0.01, 1000, 'CD4+ ({})'.format(df2.shape[0]), ha='center', **style)
        ax.text(500, 500, 'CD8+ Tcells ({})'.format(df3.shape[0]), ha='center', **style)
        
        df1.to_csv(os.path.join(current_file_path,'../data/gates/{}/cells_type_{}__CD4negCD8neg.csv'.format(data_type,cell_type['type'])), index=False, header=True)
        df2.to_csv(os.path.join(current_file_path,'../data/gates/{}/cells_type_{}__CD4pos.csv'.format(data_type,cell_type['type'])), index=False, header=True)
        df3.to_csv(os.path.join(current_file_path,'../data/gates/{}/cells_type_{}__CD8pos.csv'.format(data_type,cell_type['type'])), index=False, header=True)


    elif cell_type['type'] == 'TCRgd':
        df.plot(kind='scatter', x='152Sm_TCRgd', y='170Er_CD3', ax=ax, c='black', s=5)
        points1 = [
                    (0.1,50),
                    (8000,50),
                    (8000,8000),
                    (0.1,8000)
                    ]
        polygon1 = Polygon(points1)
        x1,y1 = polygon1.exterior.xy
        ax.plot(x1,y1, c='#d62828')

        

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

        style = dict(size=10, color='#d62828')
        ax.text(500, 10, 'TCRgd Tcells ({})'.format(df1.shape[0]), ha='center', **style)       
        

        df1.to_csv(os.path.join(current_file_path,'../data/gates/{}/cells_type_{}.csv'.format(data_type,cell_type['type'])), index=False, header=True)

    elif cell_type['type'] == 'CD8pos':
        df.plot(kind='scatter', x='146Nd_CD8a', y='143Nd_CD45RA', ax=ax, c='black', s=5)
        points1 = [
                    (0.5,0+2*eps/3),
                    (5000,0+2*eps/3),
                    (5000,10),
                    (0.5,10)
                    ]
        polygon1 = Polygon(points1)
        x1,y1 = polygon1.exterior.xy
        ax.plot(x1,y1, c='#d62828')

        points2 = [
                    (0.5,10),
                    (5000,10),
                    (5000,1000),
                    (0.5,1000)
                    ]
        polygon2 = Polygon(points2)
        x2,y2 = polygon2.exterior.xy
        ax.plot(x2,y2, c='#d62828')

      

        

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim((eps/2, 10*1000))
        ax.set_ylim((eps/2, 10*1000))
        ax.set_title('CD8+ Mem Tcells\nCD8+ Naive Tcells')
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
        print('CD8+ Mem Tcells = {}'.format(df1.shape[0]))
        print('CD8+ Naive Tcells = {}'.format(df2.shape[0]))
        print('Junk freq = {}'.format(df_junk.shape[0]))

        style = dict(size=10, color='#d62828')
        ax.text(0.01, 0.1, 'CD8+ Mem Tcells ({})'.format(df1.shape[0]), ha='center', **style)
        ax.text(0.01, 100, 'CD8+ Naive Tcells ({})'.format(df2.shape[0]), ha='center', **style)
        
        

        df1.to_csv(os.path.join(current_file_path,'../data/gates/{}/cells_type_{}__CD8posMem.csv'.format(data_type,cell_type['type'])), index=False, header=True)
        df2.to_csv(os.path.join(current_file_path,'../data/gates/{}/cells_type_{}__CD8posNaive.csv'.format(data_type,cell_type['type'])), index=False, header=True)

    elif cell_type['type'] == 'CD4pos':
        df.plot(kind='scatter', x='145Nd_CD4', y='143Nd_CD45RA', ax=ax, c='black', s=5)
        points1 = [
                    (0.5,0+2*eps/3),
                    (5000,0+2*eps/3),
                    (5000,10),
                    (0.5,10)
                    ]
        polygon1 = Polygon(points1)
        x1,y1 = polygon1.exterior.xy
        ax.plot(x1,y1, c='#d62828')

        points2 = [
                    (0.5,10),
                    (5000,10),
                    (5000,1000),
                    (0.5,1000)
                    ]
        polygon2 = Polygon(points2)
        x2,y2 = polygon2.exterior.xy
        ax.plot(x2,y2, c='#d62828')

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

        style = dict(size=10, color='#d62828')
        ax.text(0.01, 0.1, 'CD4+ Mem Tcells ({})'.format(df1.shape[0]), ha='center', **style)
        ax.text(0.01, 100, 'CD4+ Naive Tcells ({})'.format(df2.shape[0]), ha='center', **style)
        
        

        df1.to_csv(os.path.join(current_file_path,'../data/gates/{}/cells_type_{}__CD4posMem.csv'.format(data_type,cell_type['type'])), index=False, header=True)
        df2.to_csv(os.path.join(current_file_path,'../data/gates/{}/cells_type_{}__CD4posNaive.csv'.format(data_type,cell_type['type'])), index=False, header=True)

    elif cell_type['type'] == 'NKcells':
        df.plot(kind='scatter', x='175Lu_CD14', y='141Pr_CD7', ax=ax, c='black', s=5)
        points1 = [
                    (0+2*eps/3,0+2*eps/3),
                    (5000,0+2*eps/3),
                    (5000,10),
                    (0+2*eps/3,10)
                    ]
        polygon1 = Polygon(points1)
        x1,y1 = polygon1.exterior.xy
        ax.plot(x1,y1, c='#d62828')

        points2 = [
                    (0+2*eps/3,10),
                    (100,10),
                    (100,5000),
                    (0+2*eps/3,5000)
                    ]
        polygon2 = Polygon(points2)
        x2,y2 = polygon2.exterior.xy
        ax.plot(x2,y2, c='#d62828')
        

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

        style = dict(size=10, color='#d62828')
        ax.text(0.01, 0.01, 'MC cells ({})'.format(df1.shape[0]), ha='center', **style)
        ax.text(100, 1000, 'NK cells (CD7+) ({})'.format(df2.shape[0]), ha='center', **style)
        
        

        df1.to_csv(os.path.join(current_file_path,'../data/gates/{}/cells_type_{}__MCcells.csv'.format(data_type,cell_type['type'])), index=False, header=True)
        df2.to_csv(os.path.join(current_file_path,'../data/gates/{}/cells_type_{}__NKcellsCD7pos.csv'.format(data_type,cell_type['type'])), index=False, header=True)


    # plt.show()
    plt.savefig(os.path.join(current_file_path,'../data/gates/{}/cell_type_{}.pdf'.format(data_type, cell_type['type'])), format='pdf', dpi=600)

        


if __name__ == "__main__":
    
    # plot gated populations for a randomly selected patient

    current_file_path = os.path.abspath(os.path.dirname(__file__))
    
    sample_types = ['gt_Pre_all', 'gt_1hr_all', 'gen_train_all_models', 'gen_test_all_models']
    ##################
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
    for dt in sample_types:
        print('sample type = {}'.format(dt))
        for cell_type in gates:
            plot_gates(current_file_path, dt, cell_type)







    
    
