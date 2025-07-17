import os
import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import pearsonr
import sklearn
from sklearn import pipeline
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import random
 
 
def r2_metric(y_true, y_pred):
    """Returns the R² (coefficient of determination) score."""
    return r2_score(y_true, y_pred)

def extract_species_name(filename):
    base = os.path.basename(filename)
    parts = base.split('_')
    if len(parts) < 2:
        raise ValueError(f"Unexpected filename format: {base}")
    return "_".join(parts[:2]).lower()



## --- Import Directories --- ##

## Directories ##
five_map_dir = "/scratch/prj/lsm_zelezniak/recovered/projects/MSc_project/projects/specieslm/mappings/five_lab/"
three_map_dir = "/scratch/prj/lsm_zelezniak/recovered/projects/MSc_project/projects/specieslm/mappings/three_lab/"

output_dir = "/scratch/prj/lsm_zelezniak/recovered/projects/MSc_project/results/ridge_regression/lab"

## -- UPSTREAM REGRESSION -- ##

upstream_skipped_species = []

for filename in os.listdir(five_map_dir):

    #Import embedding df
    species = extract_species_name(filename)
    embed_df = pd.read_csv(os.path.join(five_map_dir, filename))
   # embed_df = pd.read_csv(os.path.join(five_map_dir, filename), encoding='latin1')


    #skip species with no boxcox tpms
    if 'boxcox_tpm' not in embed_df.columns:
        print(f"Skipping {species}: no 'boxcox_tpm' column found.")
        upstream_skipped_species.append({species})
        continue
    
    
    embed_df = embed_df.dropna(subset=['boxcox_tpm'])
    
    #Check for NAs
    if embed_df.shape[0] == 0:
        print(f"Skipping species {species} — no data after dropping NaNs")
        continue        


    embedding_cols = [col for col in embed_df.columns if col.startswith("embed_")]

    X = embed_df[embedding_cols] 

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) 


    # ----------------------
    # Fit a Ridge Regression model (train/test split: 80% train, 20% test)
    # ----------------------

    X = X_scaled
    y = embed_df["boxcox_tpm"]
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
 
    reg_pipeline = pipeline.make_pipeline(
        StandardScaler(),
        RidgeCV(cv=5, alphas=[10, 100, 1000])
    )
    reg_pipeline.fit(X_train, y_train)
    y_pred = reg_pipeline.predict(X_test)
    r2_val = r2_metric(y_test, y_pred)
    print("UPSTREAM: Test R² (Coefficient of Determination):", r2_val)
 
    # ----------------------
    # Save predictions and ground truth to a CSV file
    # ----------------------
    results_df = pd.DataFrame({
        "True_Value": y_test,
        "Predicted_Value": y_pred
    })

    r2_df = pd.DataFrame({
    "R2": [r2_val]  # wrap in a list to make it a single-row DataFrame
    })
    r2_df_path = f"/scratch/prj/lsm_zelezniak/recovered/projects/MSc_project/results/ridge_regression/lab/upstream_r2_{species}.csv"
    r2_df.to_csv(r2_df_path, index=False)
    
    results_csv_path = f"/scratch/prj/lsm_zelezniak/recovered/projects/MSc_project/results/ridge_regression/lab/species_lm_upstream_{species}.csv"
    results_df.to_csv(results_csv_path, index=False)
   

## -- DOWNSTREAM REGRESSION -- ##

downstream_skipped_species = []

for filename in os.listdir(three_map_dir):

    #Import embedding df
    species = extract_species_name(filename)
    embed_df = pd.read_csv(os.path.join(three_map_dir, filename))
    print(embed_df.columns)

    #skip species with no boxcox tpms
    if 'boxcox_tpm' not in embed_df.columns:
        print(f"Skipping {species}: no 'boxcox_tpm' column found.")
        downstream_skipped_species.append(embed_df)
        continue
    
    
    embed_df = embed_df.dropna(subset=['boxcox_tpm'])
    
    #Check for NAs
    if embed_df.shape[0] == 0:
        print(f"Skipping species {species} — no data after dropping NaNs")
        continue        


    embedding_cols = [col for col in embed_df.columns if col.startswith("embed_")]

    X = embed_df[embedding_cols] 

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) 


    # ----------------------
    # Fit a Ridge Regression model (train/test split: 80% train, 20% test)
    # ----------------------

    X = X_scaled
    y = embed_df["boxcox_tpm"]
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
 
    reg_pipeline = pipeline.make_pipeline(
        StandardScaler(),
        RidgeCV(cv=5, alphas=[10, 100, 1000])
    )
    reg_pipeline.fit(X_train, y_train)
    y_pred = reg_pipeline.predict(X_test)
    r2_val = r2_metric(y_test, y_pred)
    print("DOWNSTREAM: Test R² (Coefficient of Determination):", r2_val)
    
    # ----------------------
    # Save predictions and ground truth to a CSV file
    # ----------------------
    results_df = pd.DataFrame({
        "True_Value": y_test,
        "Predicted_Value": y_pred
    })
    
    r2_df = pd.DataFrame({
    "R2": [r2_val]  # wrap in a list to make it a single-row DataFrame
    })
    r2_df_path = f"/scratch/prj/lsm_zelezniak/recovered/projects/MSc_project/results/ridge_regression/lab/downstream_r2_{species}.csv"
    r2_df.to_csv(r2_df_path, index=False)
    
    results_csv_path = f"/scratch/prj/lsm_zelezniak/recovered/projects/MSc_project/results/ridge_regression/lab/species_lm_upstream_{species}.csv"
    results_df.to_csv(results_csv_path, index=False)
   


    results_csv_path = f"/scratch/prj/lsm_zelezniak/recovered/projects/MSc_project/results/ridge_regression/lab/species_lm_downstream_{species}.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"Predictions and true values saved to {results_csv_path}")
    print(results_df.head())



print ("FIVE PRIME SKIPPED:", upstream_skipped_species)
print ("THREE PRIME SKIPPED:", downstream_skipped_species)