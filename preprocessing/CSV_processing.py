import sys
import os
import glob
import pandas as pd 
import numpy as np 
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials



def download_files(local_download_path):

    https://drive.google.com/drive/folders/1r2fE_dTzsrWg25RkyvHXSpV_CZOObLKY?usp=sharing

    !pip install -U -q PyDrive

    # 1. Authenticate and create the PyDrive client.
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)

    # choose a local (colab) directory to store the data.
    try:
        os.makedirs(local_download_path)
    except: pass

    # 2. Auto-iterate using the query syntax
    #    https://developers.google.com/drive/v2/web/search-parameters
    file_list = drive.ListFile(
        {'q': "'1r2fE_dTzsrWg25RkyvHXSpV_CZOObLKY' in parents"}).GetList()  #use your own folder ID here

    for f in file_list:
        # 3. Create & download by id.
        print('title: %s, id: %s' % (f['title'], f['id']))
        fname = f['title']
        print('downloading to {}'.format(fname))
        f_ = drive.CreateFile({'id': f['id']})
        f_.GetContentFile(fname)


def slice_markers(file_path, timepoints):
    # keep indeces related to functional and phenotypic markers

    ############# get the indeces for markers
    Func_markers_file = '../data/markers/FuncInds.csv'
    Pheno_markers_file = '../data/markers/PhenoInds.csv'
    df_func_markers = pd.read_csv(Func_markers_file)
    func_markers = df_func_markers.iloc[:,[1]].values
    func_markers = func_markers.flatten()
    df_pheno_markers = pd.read_csv(Pheno_markers_file)
    pheno_markers = df_pheno_markers.iloc[:,[1]].values
    pheno_markers = pheno_markers.flatten()
    all_indeces = np.concatenate([func_markers,pheno_markers])

    ###############################################

    for timepoint in timepoints:

        print('timepoint = {}'.format(timepoint))
        ####################################
        list_of_files =  glob.glob(file_path+'Kehlet_V*_*_A_{0:}_*_Mononuclear Cells.csv'.format(timepoint))
        
        for filename in list_of_files:
    
            df = pd.read_csv(filename)
            df = df.iloc[:, all_indeces]
            os.makedirs('../data/preprocessed/{0:}/'.format(timepoint), exist_ok=True)
            if df.shape[0] >= 20000:
                df = df.sample(n=20000, random_state=42)
                df.to_csv('../data/preprocessed/{0:}/Func_Pheno_20k_'.format(timepoint)+filename.split('/')[-1][7:10]+'_A_{0:}.csv'.format(timepoint), header=True, index=False)
            else:
                df.to_csv('../data/preprocessed/{0:}/Func_Pheno_'.format(timepoint)+filename.split('/')[-1][:10]+'_A_{0:}.csv'.format(timepoint), header=True, index=False)

               
                
              



if __name__ == "__main__":
    
    filepath = '../data/raw_csv/'
    timepoints = ['Pre', '1hr']
    download_files(filepath)
    # slice_markers(filepath, timepoints)



    
    
