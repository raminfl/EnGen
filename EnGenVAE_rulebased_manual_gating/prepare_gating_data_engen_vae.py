import sys
import os
import pandas as pd 
import numpy as np 


def inv_arcsinh_transformation(x):
    a = 0
    b = 1/5
    c = 0

    return (np.sinh(x-c)-a)/b    

   

def all_generated_test(current_file_path):

    #concat gen csv

    column_labels = ['149Sm_CREB', '150Nd_STAT5', '151Eu_p38', '153Eu_STAT1', '154Sm_STAT3', '155Gd_S6', '159Tb_MAPKAPK2', '164Dy_IkB', '166Er_NFkB', '167Er_ERK', '168Er_pSTAT6', '113In_CD235ab_CD61', '115In_CD45', '143Nd_CD45RA', '139La_CD66',
                    '141Pr_CD7', '142Nd_CD19', '144Nd_CD11b', '145Nd_CD4', '146Nd_CD8a', '147Sm_CD11c', '148Nd_CD123', '156Gd_CD24', '157Gd_CD161', '158Gd_CD33', '165Ho_CD16', '169Tm_CD25', '170Er_CD3', '171Yb_CD27', '172Yb_CD15',
                    '173Yb_CCR2', '175Lu_CD14',	'176Lu_CD56', '160Gd_Tbet', '162Dy_FoxP3', '152Sm_TCRgd', '174Yb_HLADR']

    for gen_id in range(27):
        
        df_all_gen = pd.DataFrame(columns=column_labels)
        print('-------test_id = {}'.format(gen_id))
        for iter_id in range(30):
            print('---iter_id = {}'.format(iter_id))

            if iter_id < 10:
                iter_id = 'iter_0{}'.format(iter_id)
            else:
                iter_id = 'iter_{}'.format(iter_id)
            
            df_sample_gen = pd.read_csv(os.path.join(current_file_path,'../EnGenVAE_train_iterations/engen_vae_output/{0:}/generated/generated_{0:}_{1:}.csv'.format(iter_id, gen_id)))
            df_sample_gen = df_sample_gen.applymap(inv_arcsinh_transformation)
            df_sample_gen[df_sample_gen < 0] = 0
            assert len(column_labels)==len(df_sample_gen.columns.values), 'Columns do not match!'
            df_sample_gen.columns = column_labels
            df_sample_gen.reset_index(drop=True, inplace=True)
            
            df_all_gen = pd.concat([df_all_gen, df_sample_gen], axis=0)
        
        df_all_gen.reset_index(drop=True, inplace=True)
        os.makedirs(os.path.join(current_file_path,'../data/gates_engen_vae/gen_test/'), exist_ok=True)
        df_all_gen.to_csv(os.path.join(current_file_path,'../data/gates_engen_vae/gen_test/gen_test_{}_all_models_inv_scaled_inv_arcsinh.csv'.format(gen_id)), index=False, header=True)

        


if __name__ == "__main__":
    
    current_file_path = os.path.abspath(os.path.dirname(__file__))
    all_generated_test(current_file_path) # test_generated samples by all models for one patient 






    
    
