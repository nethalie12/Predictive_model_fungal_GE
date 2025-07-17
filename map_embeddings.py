#Defining directories and packages
import numpy as np
import pandas as pd
import os
import glob
import fastparquet


###

#READING IN FILES

###

def extract_species_name(filename):
    base = os.path.basename(filename)
    parts = base.split('_')
    if len(parts) < 2:
        raise ValueError(f"Unexpected filename format: {base}")
    return "_".join(parts[:2]).lower()

##Reading in the parquet files
three_parquet_files = glob.glob("/scratch/prj/lsm_zelezniak/recovered/projects/MSc_project/originals/Sequences/proj_seq/batch1/**/*three_prime.parquet", recursive=True)
five_parquet_files = glob.glob("/scratch/prj/lsm_zelezniak/recovered/projects/MSc_project/originals/Sequences/proj_seq/batch1/**/*five_prime.parquet", recursive=True)

three_parquet = [] 
five_parquet = [] 


for file_path in three_parquet_files:
    df = pd.read_parquet(file_path)
    df = df[df["three_prime_seq"].str.len() >= 1003]
    gene_ids = df["gene_id"].reset_index(drop=True)
    species = extract_species_name(file_path)
    three_parquet.append((species, gene_ids))


for file_path in five_parquet_files:
    df = pd.read_parquet(file_path)
    df = df[df["five_prime_seq"].str.len() >= 1003]
    gene_ids = df["gene_id"].reset_index(drop=True)
    species = extract_species_name(file_path)
    five_parquet.append((species, gene_ids))


print("parquet files read")

##Reading in embeddings
three_embed_dir = "/scratch/prj/lsm_zelezniak/recovered/projects/MSc_project/projects/specieslm/embeddings/three_prime_embeddings"
five_embed_dir = "/scratch/prj/lsm_zelezniak/recovered/projects/MSc_project/projects/specieslm/embeddings/five_prime_embeddings"

three_embed = []
five_embed = []

for filename in os.listdir(three_embed_dir):
    if filename.endswith("_three_embeddings.npy"):
        file_path = os.path.join(three_embed_dir, filename)
        species = extract_species_name(file_path)
        df = np.load(file_path)
        three_embed.append((species, df))


for filename in os.listdir(five_embed_dir):
    if filename.endswith("_five_embeddings.npy"):
        file_path = os.path.join(five_embed_dir, filename)
        species = extract_species_name(file_path)
        df = np.load(file_path)
        five_embed.append((species, df))

print("embeddings read")

##Reading in box_cox tpm values##

tpm_dir = "/scratch/prj/lsm_zelezniak/recovered/projects/MSc_project/results/output_files_species_fungiexp/bc_tpm"

bc_tpm =[]

for filename in os.listdir(tpm_dir):
    if filename.endswith(".tsv"):
        file_path = os.path.join(tpm_dir, filename)
        species = extract_species_name(file_path)
        df = pd.read_csv(file_path, sep="\t")
        bc_tpm.append((species, df)) 

print ("boxcox tpm files read in")

### --------

#MAPPING EMBEDDINGS

### --------

three_output_dir = "/scratch/prj/lsm_zelezniak/recovered/projects/MSc_project/projects/specieslm/mappings/map2/three_map"
five_output_dir = "/scratch/prj/lsm_zelezniak/recovered/projects/MSc_project/projects/specieslm/mappings/map2/five_map"

three_mapped_embeddings = {}
five_mapped_embeddings = {}

three_geneid_dict = {species: geneid for (species, geneid) in three_parquet} #mapping species to gene_ids for species in the parquet file
five_geneid_dict = {species: geneid for (species, geneid) in five_parquet}

print("parquet files and species dictionary created")

three_embed_dict = {species: embed for species, embed in three_embed if species in three_geneid_dict}  #dictionary of species only in both embeddings and the parquet files
five_embed_dict = {species: embed for species, embed in five_embed if species in five_geneid_dict}

print("embeddings and species dictionary created")

three_bc_tpm_dict = {species: df for (species, df) in bc_tpm if species in three_geneid_dict} #dictionary of tpm values filtered by species present in the embeddings and parquet file
five_bc_tpm_dict = {species: df for (species, df) in bc_tpm if species in five_geneid_dict}

print("filtered tpm and species dictionary created")




print("TPM THREE:", three_bc_tpm_dict.keys())
print("THREE EMBEDDINGS:", three_embed_dict.keys())
print("TPM FIVE:", five_bc_tpm_dict.keys())
print("FIVE EMBEDDINGS:", five_embed_dict.keys())





for species, embed_array in three_embed_dict.items():
    gene_ids = pd.Series(three_geneid_dict[species], name="gene_id")
    df = pd.DataFrame(embed_array, columns=[f"embed_{i}" for i in range(embed_array.shape[1])])
    df["gene_id"] = gene_ids.astype(str).str.strip()


    if species in three_bc_tpm_dict:
        tpm = three_bc_tpm_dict[species].rename(columns={"geneId": "gene_id"})
        tpm["gene_id"] = tpm["gene_id"].astype(str).str.strip()
        
        # Debug prints
        print(f"Species: {species}")
        print(f"Embedding df shape: {df.shape}")
        print(f"TPM df shape: {tpm.shape}")
        print(f"Embedding gene_id sample: {df['gene_id'].head(5).tolist()}")
        print(f"TPM gene_id sample: {tpm['gene_id'].head(5).tolist()}")
        
        common_ids = set(df["gene_id"]).intersection(set(tpm["gene_id"]))
        print(f"Matching gene_ids: {len(common_ids)}")

        df = df.merge(tpm[["gene_id","boxcox_tpm"]], on="gene_id", how="left")

    df.to_csv(os.path.join(three_output_dir, f"{species.lower()}_mapped_embeddings.csv"), index=False)





for species, embed_array in five_embed_dict.items():
    gene_ids = pd.Series(five_geneid_dict[species], name="gene_id")
    df = pd.DataFrame(embed_array, columns=[f"embed_{i}" for i in range(embed_array.shape[1])])
    df["gene_id"] = gene_ids.astype(str).str.strip()

    if species in five_bc_tpm_dict:
        tpm = five_bc_tpm_dict[species].rename(columns={"geneId": "gene_id"})
        tpm["gene_id"] = tpm["gene_id"].astype(str).str.strip()
        
        # Debug prints
        print(f"Species: {species}")
        print(f"Embedding df shape: {df.shape}")
        print(f"TPM df shape: {tpm.shape}")
        print(f"Embedding gene_id sample: {df['gene_id'].head(5).tolist()}")
        print(f"TPM gene_id sample: {tpm['gene_id'].head(5).tolist()}")
        
        common_ids = set(df["gene_id"]).intersection(set(tpm["gene_id"]))
        print(f"Matching gene_ids: {len(common_ids)}")

        df = df.merge(tpm[["gene_id","boxcox_tpm"]], on="gene_id", how="left")

    df.to_csv(os.path.join(five_output_dir, f"{species.lower()}_mapped_embeddings.csv"), index=False)




