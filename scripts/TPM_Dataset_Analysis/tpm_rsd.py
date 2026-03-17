import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import gffutils
import os
from Bio import SeqIO
import pandas as pd
import re
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from rnanorm.datasets import load_toy_data
from rnanorm import TPM
from scipy import stats
from pathlib import Path
import glob
from tpm_rsd_functions import rsd, filter_tpm_rsd_dataset, filter_tpm_rsd_dataset_geneId, transform_tpm, plot_transformed


mapping_df = pd.read_csv("../output_with_species.csv")

taxon_mapping = dict(zip(mapping_df['Taxon_ID'].astype(str), mapping_df['Species_name']))
all_dataframes = []
base_path = "../fungiexp_data/"
taxon_folders = [f for f in os.listdir(base_path) if f.isdigit()] # Get all taxon folders in the base_path (getting directories starting with a number)

###  Loop through each taxon folder and read the TSV files with the TPM values 
for taxon in taxon_folders:
    folder_path = os.path.join(base_path, taxon)
    files = glob.glob(os.path.join(folder_path, "*.tsv"))

    for file_path in files:
        df = pd.read_csv(file_path, sep="\t", dtype=str)
        df = df[['geneId', 'TPM']]
        df['taxon'] = taxon
        all_dataframes.append(df)

###  Combine all the dataframes into a single DataFrame
combined_df = pd.concat(all_dataframes, ignore_index=True)

### Filter out rows with TPM <= 0 and save the resulting DataFrame
combined_df['TPM'] = pd.to_numeric(combined_df['TPM'], errors='coerce')
filtered_df = combined_df[combined_df['TPM'] > 0].copy()
filtered_df['taxon'] = filtered_df['taxon'].map(taxon_mapping).fillna('')
filtered_df.to_csv("../filtered_df_fungiexp_data.csv", index=False)

###  Save the resulting DataFrame as a CSV file for each species
for taxon, species_name in taxon_mapping.items():
    species_df = filtered_df[filtered_df['taxon'] == species_name]
    if not species_df.empty:
        species_name_formatted = species_name.replace(" ", "_")
        species_df.to_csv(f"../filtered_df_{species_name_formatted}.csv", index=False)
        print(f"{species_name} {species_df.shape}")

# Print the result
print(filtered_df) 

# Load taxon-to-species mapping
# -------------------------
mapping_df = pd.read_csv("../output_with_species.csv")
taxon_mapping = dict(zip(mapping_df['Taxon_ID'].astype(str), mapping_df['Species_name']))

# -------------------------
# Define threshold combinations to test
# -------------------------
TPM_thresholds = [1, 5]
RSD_thresholds = [1, 2]

# -------------------------
# Loop through each species file and perform analysis
# -------------------------
for taxon, species_name in taxon_mapping.items():
    species_name_formatted = species_name.replace(" ", "_")
    file_path = f"../filtered_df_{species_name_formatted}.csv"
    
    if os.path.exists(file_path):
        # Load and clean the species-specific dataset
        filtered_df = pd.read_csv(file_path, sep=",")
        filtered_df = filtered_df.dropna()
        filtered_df = filtered_df.replace([np.inf, -np.inf], np.nan).dropna()
        
        print(f"\nProcessing {species_name} with shape {filtered_df.shape}")
        print(filtered_df.head())
        
        # Ensure output directory exists
        Path(".././output_files_species_fungiexp").mkdir(parents=True, exist_ok=True)

        # Exploratory plotting of RSD metrics across a range of TPM thresholds
        dfs = []
        for i in range(0, 6):
            desc = (filtered_df
                    .groupby('geneId')
                    .agg({'TPM': ['median', 'std', rsd]})["TPM"]
                    .query('median > {}'.format(i))
                    .rsd
                    .describe(percentiles=[.5, .6, .7, .75, .8, .9, .95, .99])
                   ).reset_index()
            desc["tmp_threshold"] = i
            dfs.append(desc)
            print('TPM > {}'.format(i))
        tmp_thresholds = pd.concat(dfs)
        toPlot = tmp_thresholds[tmp_thresholds["index"].isin(["50%", "60%", "70%", "75%", "80%", "90%", "95%", "99%"])]
        print(toPlot)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=False, sharey=False)
        sns.set_theme()
        sns.stripplot(ax=axes[0], y="rsd", x="index", hue="tmp_threshold", data=toPlot).set(xlabel="RSD percentile", ylabel="RSD")
        sns.stripplot(ax=axes[1], y="rsd", x="tmp_threshold", data=tmp_thresholds.query('index == "count"')).set(xlabel="TPM threshold", ylabel="Dataset size")
        for ax in axes:
            ax.grid(False)
            ax.set_facecolor('none')
        fig.patch.set_alpha(0.0)
      
        fig.savefig(f".././output_files_species_fungiexp/{species_name_formatted}_rsd_tpm_thresholds.png", dpi=300)
 
        # -------------------------
        # Iterate over threshold combinations to choose the best one
        # -------------------------
        best_score = np.inf
        best_thresholds = None
        best_transformed_df = None
        
        for tpm_thresh in TPM_thresholds:
            for rsd_thresh in RSD_thresholds:
                # Apply aggregated filtering
                temp_df = filter_tpm_rsd_dataset(filtered_df, tpm_thresh, rsd_thresh)
                if temp_df.empty:
                    continue
                # Transform the filtered data
                temp_df = transform_tpm(temp_df)

                # Compute normalization metrics using Box-Cox if available, otherwise Yeo-Johnson
                if "boxcox_tpm" in temp_df.columns and temp_df["boxcox_tpm"].notna().all():
                    skew_val = abs(stats.skew(temp_df["boxcox_tpm"]))
                    kurt_val = abs(stats.kurtosis(temp_df["boxcox_tpm"], fisher=True))
                else:
                    skew_val = abs(stats.skew(temp_df["yj_tpm"]))
                    kurt_val = abs(stats.kurtosis(temp_df["yj_tpm"], fisher=True))
                score = skew_val + kurt_val
                print(f"For {species_name}: TPM > {tpm_thresh}, RSD < {rsd_thresh} -> skew: {skew_val:.4f}, kurtosis: {kurt_val:.4f}, score: {score:.4f}")
                
                if score < best_score:
                    best_score = score
                    best_thresholds = (tpm_thresh, rsd_thresh)
                    best_transformed_df = temp_df.copy()
                    
        if best_thresholds is None:
            print(f"No valid threshold combination found for {species_name}. Skipping.")
            continue
            
        best_tpm_thresh, best_rsd_thresh = best_thresholds

        # -------------------------
        # Save aggregated/transformed output and plots
        # -------------------------
        agg_output_filename = f".././output_files_species_fungiexp/{species_name_formatted}_tpm_{best_tpm_thresh}_rsd_{best_rsd_thresh}.tsv"
        best_transformed_df.to_csv(agg_output_filename, sep="\t", index=False)
        print(f"Aggregated: Selected thresholds for {species_name}: TPM > {best_tpm_thresh}, RSD < {best_rsd_thresh} with score: {best_score:.4f}")
        
        # Generate and save histogram plots for the aggregated output
        plot_transformed(best_transformed_df, f"TPM > {best_tpm_thresh}, RSD < {best_rsd_thresh}", agg_output_filename)
        
        # Plot and save the quantile–quantile plot for Box-Cox TPM if available
        if "boxcox_tpm" in best_transformed_df.columns and best_transformed_df["boxcox_tpm"].notna().all():
            plt.figure(figsize=(10, 6))
            qq_array = stats.probplot(best_transformed_df["boxcox_tpm"], dist="norm", plot=None)
            x, y = qq_array[0]
            slope, intercept, r = qq_array[1]
            plt.plot(x, y, 'o')
            plt.plot(x, slope * x + intercept, 'r', lw=2)
            plt.title(f'{species_name} - Box-Cox TPM\nR-squared: {r:.4f}')
            plt.xlabel('Theoretical Quantiles')
            plt.ylabel('Ordered Values')
            qq_png_path = agg_output_filename.replace(".tsv", "_qq.png")
            plt.savefig(qq_png_path, dpi=300)
        
        # -------------------------
        # Use geneId-level filtering (ordered dataset) with the best thresholds
        # -------------------------
        ordered_filtered_df = filter_tpm_rsd_dataset_geneId(filtered_df, best_tpm_thresh, best_rsd_thresh)
        ordered_output_filename = f".././output_files_species_fungiexp/{species_name_formatted}_tpm_{best_tpm_thresh}_rsd_{best_rsd_thresh}_ordered.tsv"
        ordered_filtered_df.to_csv(ordered_output_filename, sep="\t", index=False)
        print(f"Ordered output saved for {species_name}")
