# 

# EnGen: In-Silico Generation of High-Dimensional Immune Response in Patients using A Deep Neural Network

Link to paper: [link] \
Codes and data made available for easy retraining of models, reproduction of results, and plots presented by the authors in the main text and supplementary materials.

## Downloding data, cleaning, and preprocessing
```
python3 preprocessing/CSV_processing.py
```
## Visualizations for the raw data samples
```
python3 raw_data_plots/visualize_raw_data.py
```
## preprocessing of data for iterations of EnGen training
```
python3 EnGen_iteration_preprocess.py
```
## training EnGen models per iteration
```
python3 EnGen_model/train.py --iter_id $ITERATION_ID_ZERO_INDEXED
```
## generate samples from saved models 
```
python3 EnGen_generate/generate_train.py
python3 EnGen_generate/generate_test.py
```
## PCA plots of ground-truth and generated samples from saved models 
```
python3 EnGen_generate/plot_source_target_samples.py
```
## perform rulebased manaul gating on ground-truth and generated samples and make plots
```
python3 Rulebased_manual_gating/prepare_gating_data.py
python3 Rulebased_manual_gating/plot_raw_donut_PCA_colored_by_cell_type.py
python3 Rulebased_manual_gating/plot_gated_random_patient.py
python3 Rulebased_manual_gating/plot_gated_all_patients.py
python3 Rulebased_manual_gating/compare_gates.py
```
## perform classification experiements and plot results
```
python3 Classification/train_random_forest.py
python3 Classification/plot_RF_scores.py
```

## citation information
Please cite the paper if you use any part of the code or datasets.
[citation]

## prerequisites
```
Platform: x86_64-pc-linux-gnu (64-bit)
Running under: Ubuntu 18.04.3 LTS
python version 3.6.9 [GCC 8.4.0]
CUDA version v9.2
python libraries:
  scipy 1.5.4
  pandas 0.24.2
  numpy 1.19.5
  sklearn 0.24.1
  torch 1.2.0+cu92
  matplotlib 3.1.0
  seaborn 0.9.0
  plotly 4.14.1
  shapely 1.7.1
  tqdm 4.33.0
  progressbar 2.5
  oauth2client 4.1.3
  
```
