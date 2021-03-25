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
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial import distance
from sklearn.manifold import TSNE
from scipy.optimize import linear_sum_assignment
import squarify
import plotly.graph_objects as go

def inv_arcsinh_transformation(x):
    a = 0
    b = 1/5
    c = 0
    # if np.isinf((np.sinh(x-c)-a)/b):
    #     print(x)
    #     print((np.sinh(x-c)-a)/b)
        # sys.exit()
    return (np.sinh(x-c)-a)/b


def one_gt_matched_sankey_colored_by_kmeans(): 
    # cluster Pre and 1hr matched gt by kmeans from a patient

    random.seed(42)
    all_patients = pickle.load(open('/home/raminf/DL/EnGen/FCS_files_kehlet/iter_33pc_train/preprocessed_data/all_patient_ids.pkl', 'rb'))['all_patient_ids']
    column_labels = ['149Sm_CREB', '150Nd_STAT5', '151Eu_p38', '153Eu_STAT1', '154Sm_STAT3', '155Gd_S6', '159Tb_MAPKAPK2', '164Dy_IkB', '166Er_NFkB', '167Er_ERK', '168Er_pSTAT6', '113In_CD235ab_CD61', '115In_CD45', '143Nd_CD45RA', '139La_CD66',
                    '141Pr_CD7', '142Nd_CD19', '144Nd_CD11b', '145Nd_CD4', '146Nd_CD8a', '147Sm_CD11c', '148Nd_CD123', '156Gd_CD24', '157Gd_CD161', '158Gd_CD33', '165Ho_CD16', '169Tm_CD25', '170Er_CD3', '171Yb_CD27', '172Yb_CD15',
                    '173Yb_CCR2', '175Lu_CD14',	'176Lu_CD56', '160Gd_Tbet', '162Dy_FoxP3', '152Sm_TCRgd', '174Yb_HLADR']
#     df_all_gt_Pre_1hr = pd.read_csv('/home/raminf/DL/EnGen/FCS_files_kehlet/iter_33pc_train/plots/gates/gt_Pre_1hr_all_patients.csv')
#     df_all_gt_Pre =  df_all_gt_Pre_1hr[df_all_gt_Pre_1hr['timepoint']=='Pre']
#     df_all_gt_1hr =  df_all_gt_Pre_1hr[df_all_gt_Pre_1hr['timepoint']=='1hr']
#     df_all_gt_Pre_1hr.drop(['timepoint', 'patient_id'], axis=1, inplace=True)
#     df_all_gt_Pre_1hr = df_all_gt_Pre_1hr.sample(n=20000, random_state=42)
#     df_all_gt_Pre_1hr.reset_index(drop=True, inplace=True)
#     scaler = StandardScaler()
    # scaler.fit(df_all_gt_Pre.values) # fit on Pre only
#     scaler.fit(df_all_gt_Pre_1hr.values) # fit on Pre and 1hr
#     df_all_gt_Pre_1hr.loc[:,:] = scaler.transform(df_all_gt_Pre_1hr.values)
#     pca = PCA(n_components=2, whiten=False)
    # pca.fit(df_all_gt_Pre) # fit on Pre only
#     pca.fit(df_all_gt_Pre_1hr) # fit on Pre and 1hr
#     df_pca_Pre_1hr = pd.DataFrame(data=pca.transform(df_all_gt_Pre_1hr), columns=['PC1', 'PC2'])

#     kmeans = MiniBatchKMeans(n_clusters=3, random_state=42, batch_size=2000).fit(df_all_gt_Pre_1hr)

    kmeans_pickle = pickle.load(open('/home/raminf/DL/EnGen/FCS_files_kehlet/iter_33pc_train/plots/Sankey_matching_Pre_1hr_colored/Donut_plot_scaler_kmeans.pkl', 'rb'))
    scaler = kmeans_pickle['scaler']
    kmeans = kmeans_pickle['kmeans']
    
    plot_patient = random.sample(all_patients, 1)[0] # select one patient randomly for plotting in Fig2 hexbin
    print('plot_patient = {}'.format(plot_patient))
    
    for patient_id in [plot_patient]: # only generate the plot_patient
    # for patient_id in ['V07', 'V53', 'V60', 'V04', 'V46', 'V47', 'V06', 'V52', 'V02']: # only generate the plot_patient
    
        # df_gt_Pre_p =  df_all_gt_Pre[df_all_gt_Pre['patient_id']==patient_id]
        # df_gt_1hr_p =  df_all_gt_1hr[df_all_gt_1hr['patient_id']==patient_id]
        
        # df_gt_Pre_p.drop(['timepoint', 'patient_id'], axis=1, inplace=True)
        # df_gt_1hr_p.drop(['timepoint', 'patient_id'], axis=1, inplace=True)  
        
        # df_gt_Pre_p = df_gt_Pre_p.sample(frac=1, random_state=42)
        # df_gt_1hr_p = df_gt_1hr_p.sample(frac=1, random_state=42)

        # df_gt_Pre_p.loc[:,:] = scaler.transform(df_gt_Pre_p.values)
        # df_gt_1hr_p.loc[:,:] = scaler.transform(df_gt_1hr_p.values)
        
        # df_gt_Pre_p.reset_index(drop=True, inplace=True)
        # df_gt_1hr_p.reset_index(drop=True, inplace=True)

        df_Pre_1hr_matched_p = pd.read_csv('/home/raminf/DL/EnGen/FCS_files_kehlet/iter_33pc_train/preprocessed_data/run_0/iter_00/Func_Pheno_45k_scaled_with_Pre_1hr_tps_source_Pre_target_1hr_matched.csv')
        df_Pre_1hr_matched_p = df_Pre_1hr_matched_p[df_Pre_1hr_matched_p['patient_id']==patient_id]
        df_Pre_1hr_matched_p.drop('patient_id', axis=1, inplace=True)

        

        df_Pre_1hr_matched_p.reset_index(drop=True, inplace=True)
        # print(df_gt_Pre_p.describe())
        # print(df_gt_1hr_p.describe())
        # print(df_Pre_1hr_matched_p.describe())
        # df_sample_gen = df_sample_gen.applymap(inv_arcsinh_transformation) # already done
        # df_sample_gen[df_sample_gen < 0] = 0 # already done
        assert len(column_labels)*2==len(df_Pre_1hr_matched_p.columns.values), 'Column sizes do not match!'
        # df_sample_gen.columns = column_labels
        df_Pre_matched_p = df_Pre_1hr_matched_p.iloc[:,:len(column_labels)]
        df_Pre_matched_p.columns = column_labels
        df_1hr_matched_p = df_Pre_1hr_matched_p.iloc[:,len(column_labels):]
        df_1hr_matched_p.columns = column_labels
        # df_Pre_matched_p.iloc[:,:] = scaler.transform(df_Pre_matched_p.values)
        # df_1hr_matched_p.iloc[:,:] = scaler.transform(df_1hr_matched_p.values)
        df_p = pd.concat([df_Pre_matched_p, df_1hr_matched_p], axis=0)
        # kmeans = MiniBatchKMeans(n_clusters=3, random_state=42, batch_size=2000).fit(df_p)

        df_Pre_1hr_matched_p['cluster_id_Pre'] = kmeans.predict(df_Pre_1hr_matched_p.iloc[:,:len(column_labels)])
        df_Pre_1hr_matched_p['cluster_id_1hr'] = kmeans.predict(df_Pre_1hr_matched_p.iloc[:,len(column_labels):-1]) # excluding the cluster_id_Pre column that was just added
        print('Pre 0')
        print(df_Pre_1hr_matched_p[df_Pre_1hr_matched_p['cluster_id_Pre']==0].shape[0])
        print('1hr 0')
        print(df_Pre_1hr_matched_p[df_Pre_1hr_matched_p['cluster_id_1hr']==0].shape[0])
        print('Pre 1')
        print(df_Pre_1hr_matched_p[df_Pre_1hr_matched_p['cluster_id_Pre']==1].shape[0])
        print('1hr 1')
        print(df_Pre_1hr_matched_p[df_Pre_1hr_matched_p['cluster_id_1hr']==1].shape[0])
        print('Pre 2')
        print(df_Pre_1hr_matched_p[df_Pre_1hr_matched_p['cluster_id_Pre']==2].shape[0])
        print('1hr 2')
        print(df_Pre_1hr_matched_p[df_Pre_1hr_matched_p['cluster_id_1hr']==2].shape[0])

        print(df_Pre_1hr_matched_p[df_Pre_1hr_matched_p['cluster_id_Pre']==df_Pre_1hr_matched_p['cluster_id_1hr']].shape[0]/df_Pre_1hr_matched_p.shape[0])

        colors=['#81ecec', '#74b9ff', '#a29bfe']
        hex_color = [i.lstrip("#") for i in colors]
        colors = ['rgba({}, {}, {}, {})'.format(int(i[0:2], 16), int(i[2:4], 16), int(i[4:6], 16), 0.2) for i in hex_color]
        label = ['', '', '', '', '', '']
        node_colors = ['#81ecec', '#74b9ff', '#a29bfe', '#81ecec', '#74b9ff', '#a29bfe']
        source = []
        target = []
        value = []
        link_colors = []
        for id_Pre in range(3):
            for id_1hr in range(3,7):
                df_node = df_Pre_1hr_matched_p[df_Pre_1hr_matched_p['cluster_id_Pre']==id_Pre]
                source.append(id_Pre)
                target.append(id_1hr)
                value.append(df_node[df_node['cluster_id_1hr']==id_1hr-3].shape[0]/df_Pre_1hr_matched_p.shape[0]*100)
                link_colors.append(colors[id_Pre])
        print(source)
        print(target)
        print(value)
        link = dict(source = source, target = target, value = value, color=link_colors)
        node = dict(label = label, pad=50, thickness=15, color=node_colors)
        data = go.Sankey(link = link, node=node)
        # plot using plotly instead of matplotlib
        fig = go.Figure(data)
        fig.update_layout(autosize=False, width=500, height=500)
        fig.write_image('/home/raminf/DL/EnGen/FCS_files_kehlet/iter_33pc_train/plots/Sankey_matching_Pre_1hr_colored/Sankey_plot_matching_{}.png'.format(patient_id))
        # fig.show()
        



if __name__ == "__main__":
    

    
    one_gt_matched_sankey_colored_by_kmeans()






    
    
