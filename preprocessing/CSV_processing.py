import sys
import os
import glob
import pandas as pd 
import numpy as np 
import requests
from apiclient import discovery
import oauth2client
from oauth2client import client
from oauth2client import tools
import progressbar
import zipfile

 
def download_files(current_file_path):
#modified https://github.com/nsadawi/Download-Large-File-From-Google-Drive-Using-Python

    destination = os.path.join(current_file_path,'../data/raw_csv/')
    os.makedirs(destination, exist_ok=True)

    URL = "https://docs.google.com/uc?export=download"
    id = '1iTYoUroklbYqamr-oOZuNhz84BdBwvhu'
    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)
        

    destination = destination+'raw_csv.zip'
    save_response_content(response, destination) 
    unzip(destination)   

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    total_size = len(response.content)

    with open(destination, "wb") as f:
        print('downloading.. It may take a few minutes.')
        pbar = progressbar.ProgressBar(maxval=total_size//CHUNK_SIZE)
        for chunk in pbar(response.iter_content(CHUNK_SIZE)):
            
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def unzip(path_to_zip_file):

    directory_to_extract_to = path_to_zip_file.split('raw_csv.zip')[0]
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)
    os.remove(path_to_zip_file)

def slice_markers(current_file_path, timepoints):
    # keep indeces related to functional and phenotypic markers
    ############# get the indeces for markers
    data_file_path = os.path.join(current_file_path,'../data/raw_csv/')
    markers_file_path = os.path.join(current_file_path,'../data/markers/')
    Func_markers_file = markers_file_path+'FuncInds.csv'
    Pheno_markers_file = markers_file_path+'PhenoInds.csv'
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
        list_of_files =  glob.glob(data_file_path+'Kehlet_V*_*_A_{0:}_*_Mononuclear Cells.csv'.format(timepoint))
        
        save_path = os.path.join(current_file_path,'../data/preprocessed/{0:}/'.format(timepoint))
        os.makedirs(save_path, exist_ok=True)
        for filename in list_of_files:
    
            df = pd.read_csv(filename)
            df = df.iloc[:, all_indeces]
            
            if df.shape[0] >= 20000:
                df = df.sample(n=20000, random_state=42)
                df.to_csv(save_path+'Func_Pheno_20k_'.format(timepoint)+filename.split('/')[-1][7:10]+'_A_{0:}.csv'.format(timepoint), header=True, index=False)
            else:
                df.to_csv(save_path+'Func_Pheno_'.format(timepoint)+filename.split('/')[-1][:10]+'_A_{0:}.csv'.format(timepoint), header=True, index=False)

          
                
              



if __name__ == "__main__":
    
    current_file_path = os.path.abspath(os.path.dirname(__file__))
    timepoints = ['Pre', '1hr']
    download_files(current_file_path)
    slice_markers(current_file_path, timepoints)



    
    
