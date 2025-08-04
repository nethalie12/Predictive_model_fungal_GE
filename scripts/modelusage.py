import math
import itertools
import collections
from collections.abc import Mapping
import numpy as np
import pandas as pd
import tqdm
import os
import torch
import glob
import fastparquet
from torch.utils.data import DataLoader
from functools import partial

from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding

from datasets import Dataset

print("All dependencies are imported")

##Utilities
def chunkstring(string, length):
    # chunks a string into segments of length
    return (string[0+i:length+i] for i in range(0, len(string), length))

def kmers(seq, k=6):
    # splits a sequence into non-overlappnig k-mers
    return [seq[i:i + k] for i in range(0, len(seq), k) if i + k <= len(seq)]

def kmers_stride1(seq, k=6):
    # splits a sequence into overlapping k-mers
    return [seq[i:i + k] for i in range(0, len(seq)-k+1)]   

def one_hot_encode(gts, dim=5):
    # one-hot encodes the sequence
    result = []
    nuc_dict = {"A":0,"C":1,"G":2,"T":3}
    for nt in gts:
        vec = np.zeros(dim)
        vec[nuc_dict[nt]] = 1
        result.append(vec)
    return np.stack(result, axis=0)

def class_label_gts(gts):
    # make labels into ground truths
    nuc_dict = {"A":0,"C":1,"G":2,"T":3}
    return np.array([nuc_dict[x] for x in gts])

def tok_func_standard(x, five_seq_col, three_seq_col): return tokenizer(" ".join(kmers_stride1(x[five_seq_col, three_seq_col])))


def count_special_tokens(tokens, tokenizer, where = "left"):
    count = 0
    if where == "right":
        tokens = tokens[::-1]
    for pos in range(len(tokens)):
        tok = tokens[pos]
        if tok in tokenizer.all_special_ids:
            count += 1
        else:
            break
    return count   


# def tok_func_species(example, tokenizer, seq_col):
#     seq = example[seq_col]
#     species_str = example["species_proxy"]
#     kmer_seq = " ".join(kmers_stride1(seq))
#     full_str = species_str + " " + kmer_seq
#     return tokenizer(full_str, padding="max_length", truncation=True)

def tok_func_species(batch, tokenizer, seq_col):
    kmers_list = [" ".join(kmers_stride1(seq)) for seq in batch[seq_col]]
    species_list = batch["species_proxy"]
    full_strs = [species + " " + kmers for species, kmers in zip(species_list, kmers_list)]
    return tokenizer(full_strs, padding="max_length", truncation=True, max_length=512)

##PARAMETERS

# Find parquet files
#seq_five_prime = glob.glob("/scratch/prj/lsm_zelezniak/recovered/projects/MSc_project/originals/Sequences/proj_seq/lab_species/**/*five_prime.parquet", recursive=True)
seq_three_prime = glob.glob("/scratch/prj/lsm_zelezniak/recovered/projects/MSc_project/originals/Sequences/proj_seq/lab_species/**/*three_prime.parquet", recursive=True)

# seq_three_prime = glob.glob("/scratch/prj/lsm_zelezniak/recovered/projects/MSc_project/originals/Sequences/proj_seq/yarrowia_lipolytica/yarrowia_lipolytica_three_prime_cut_length_300.parquet", recursive=True)
# seq_five_prime = glob.glob("/scratch/prj/lsm_zelezniak/recovered/projects/MSc_project/originals/Sequences/proj_seq/yarrowia_lipolytica/yarrowia_lipolytica_five_prime_cut_length_300.parquet", recursive=True)


#Defines columns containing the sequences
# five_seq_col = "five_prime_seq" 
three_seq_col = "three_prime_seq"

#Defines the k-mer size for tokenisation
kmer_size = 6 # size of kmers, always 6

#Defines the species token to use
# five_proxy_species = ["_".join(os.path.basename(path).split("_")[:2]) for path in seq_five_prime]
# five_species_proxy = str(five_proxy_species) 
# print(len(five_proxy_species))

three_proxy_species = ["_".join(os.path.basename(path).split("_")[:2]) for path in seq_three_prime]
three_species_proxy = str(three_proxy_species) 
print(len(three_proxy_species))

pred_batch_size = 128*3 # batch size for rolling masking.  Rolling masking applied masked language modelling to every k-mer one at a time across the sequence
target_layer = (8,) # what hidden layers to use for embedding

print("Parameters are set")

##LOAD THE DATA AND MODEL

#load the model
from transformers import Trainer
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig  
tokenizer = AutoTokenizer.from_pretrained("gagneurlab/SpeciesLM", revision = "downstream_species_lm")
model = AutoModelForMaskedLM.from_pretrained("gagneurlab/SpeciesLM", revision = "downstream_species_lm") 


device = "cpu"
tokenized_datasets = [] # Store the tokenised datasets


#model.to(torch.bfloat16).to(device)
#model.to(torch.float16).to(device)
model.to(device)
model.eval()

print("The model is loaded")

## PREPARE THE DATA
# five_prime_dfs = []  #Store the five prime sequences

# for path in seq_five_prime:
#     df = pd.read_parquet(path, engine="pyarrow")
#     species = "_".join(os.path.basename(path).split("_")[:2])
#     df["species_proxy"] = species
#     five_prime_dfs.append(df)

# five_prime_dataset = pd.concat(five_prime_dfs, ignore_index=True)


three_prime_dfs = []

for path in seq_three_prime:
    df = pd.read_parquet(path, engine="pyarrow")
    species = "_".join(os.path.basename(path).split("_")[:2])
    df["species_proxy"] = species
    three_prime_dfs.append(df)

three_prime_dataset = pd.concat(three_prime_dfs, ignore_index=True)

print("Five prime & Three prime datasets are read")

#Create sequences with 1003 base pairs
#five_prime_dataset[five_seq_col] = five_prime_dataset[five_seq_col].str[:1003]
three_prime_dataset[three_seq_col] = three_prime_dataset[three_seq_col].str[:300] # truncate longer sequences

#five_prime_dataset = five_prime_dataset.loc[five_prime_dataset[five_seq_col].str.len() == 1003] # throw out too short sequences
three_prime_dataset = three_prime_dataset.loc[three_prime_dataset[three_seq_col].str.len() == 300]

#Create the datasets from HuggingFace
#five_ds = Dataset.from_pandas(five_prime_dataset[[five_seq_col, "species_proxy"]])
three_ds = Dataset.from_pandas(three_prime_dataset[[three_seq_col, "species_proxy"]])

#Tokenise the datasets
# five_tok_func = partial(tok_func_species, tokenizer=tokenizer, seq_col=five_seq_col)
# five_tok_ds = five_ds.map(five_tok_func, batched=True, batch_size=1000, num_proc=4)
# five_rem_species = five_tok_ds.remove_columns(["species_proxy"])

three_tok_func = partial(tok_func_species, tokenizer=tokenizer, seq_col=three_seq_col)
three_tok_ds = three_ds.map(three_tok_func, batched=True, batch_size=1000, num_proc=4)
three_rem_species = three_tok_ds.remove_columns(["species_proxy", "three_prime_seq"])

print("Sequences tokenised")


print("Raw sequence strings removed")

#Set the engine type
#five_rem_species.set_format(type="torch") 
three_rem_species.set_format(type="torch")  

#Load the data
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

#five_data_loader = DataLoader(five_rem_species, batch_size=50, collate_fn=data_collator, shuffle=False)
three_data_loader = DataLoader(three_rem_species, batch_size=50, collate_fn=data_collator, shuffle = False)
print("Data loaded")


## EMBEDDING SEQUENCES ##

def five_embed_on_batch(tokenized_data, dataset, seq_idx, 
                   special_token_offset,
                   target_layer = target_layer):
    model_input_unaltered = tokenized_data['input_ids'].clone()
    
    five_label = five_prime_dataset.iloc[seq_idx][five_seq_col]
    five_label_len = len(five_label)

    
    if five_label_len < 6:
        print("This should not occur")
        return torch.zeros(five_label_len,five_label_len,768)
    else:
        res = tokenized_data['input_ids'].clone()
        attention_mask = tokenized_data['attention_mask'].to(device)
        res = res.to(device)
        with torch.no_grad():
            embedding = model(res, attention_mask=attention_mask, output_hidden_states=True)['hidden_states'] 
    if isinstance(target_layer, int):    
        embedding = embedding[target_layer]
    elif len(target_layer) == 1:
        embedding = torch.stack(embedding[target_layer[0]:],axis=0)
        embedding = torch.mean(embedding, axis=0)
    else:
        embedding = torch.stack(embedding[target_layer[0]:target_layer[1]],axis=0)
        embedding = torch.mean(embedding, axis=0)   
    five_embedding = embedding.detach().cpu().numpy() 
    return five_embedding

def three_embed_on_batch(tokenized_data, dataset, seq_idx, 
                   special_token_offset,
                   target_layer = target_layer):
    model_input_unaltered = tokenized_data['input_ids'].clone()
    
    three_label = three_prime_dataset.iloc[seq_idx][three_seq_col]
    three_label_len = len(three_label)

    
    if three_label_len < 6:
        print("This should not occur")
        return torch.zeros(three_label_len,three_label_len,768)
    else:
        res = tokenized_data['input_ids'].clone()
        attention_mask = tokenized_data['attention_mask'].to(device)
        res = res.to(device)
        with torch.no_grad():
            embedding = model(res, attention_mask=attention_mask, output_hidden_states=True)['hidden_states'] 
    if isinstance(target_layer, int):    
        embedding = embedding[target_layer]
    elif len(target_layer) == 1:
        embedding = torch.stack(embedding[target_layer[0]:],axis=0)
        embedding = torch.mean(embedding, axis=0)
    else:
        embedding = torch.stack(embedding[target_layer[0]:target_layer[1]],axis=0)
        embedding = torch.mean(embedding, axis=0)   
    three_embedding = embedding.detach().cpu().numpy() 
    return three_embedding
     
def extract_embedding_from_pred(hidden_states, batch_pos):   
    pred_pos_min = min(max(pos - 5, 0), hidden_states.shape[1]-1)
    pred_pos_max = min(max(pos, 0), hidden_states.shape[1]-1)
    token_embedding = hidden_states[batch_pos, pred_pos_min:pred_pos_max+1, :]
    token_embedding = token_embedding.mean(axis=0)
    return token_embedding

#RUN INFERENCE
k = 6
#five_averaged_embeddings = []
three_averaged_embeddings = []
#print (dataset.iloc[0]['seq_chunked'])


# for tokenized_data in tqdm.tqdm(five_data_loader):
#     input_ids = tokenized_data['input_ids'].to(device)
#     attention_mask = tokenized_data['attention_mask'].to(device)

#     with torch.no_grad():
#         hidden_states = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)['hidden_states']

#     if isinstance(target_layer, int):
#         embedding = hidden_states[target_layer]
#     elif len(target_layer) == 1:
#         embedding = torch.stack(hidden_states[target_layer[0]:], dim=0).mean(dim=0)
#     else:
#         embedding = torch.stack(hidden_states[target_layer[0]:target_layer[1]], dim=0).mean(dim=0)

#     avg = embedding.mean(dim=1).cpu().numpy()  # average over tokens
#     five_averaged_embeddings.extend(avg)

# five_embeddings = np.stack(five_averaged_embeddings)


for tokenized_data in tqdm.tqdm(three_data_loader):
    input_ids = tokenized_data['input_ids'].to(device)
    attention_mask = tokenized_data['attention_mask'].to(device)

    with torch.no_grad():
        hidden_states = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)['hidden_states']

    if isinstance(target_layer, int):
        embedding = hidden_states[target_layer]
    elif len(target_layer) == 1:
        embedding = torch.stack(hidden_states[target_layer[0]:], dim=0).mean(dim=0)
    else:
        embedding = torch.stack(hidden_states[target_layer[0]:target_layer[1]], dim=0).mean(dim=0)

    avg = embedding.mean(dim=1).cpu().numpy()  # average over tokens
    three_averaged_embeddings.extend(avg)

three_embeddings = np.stack(three_averaged_embeddings)

### Saving embedddings for each species ###

output_dir = "/scratch/prj/lsm_zelezniak/recovered/projects/MSc_project/scripts/specieslm/"

# Set directory for saving per-species embeddings
#five_out_dir = os.path.join(output_dir, "five_prime_embeddings")
three_out_dir = os.path.join(output_dir, "three_prime_embeddings")

#os.makedirs(five_out_dir, exist_ok=True)
os.makedirs(three_out_dir, exist_ok=True)

# Convert lists to arrays
#five_embeddings = np.stack(five_averaged_embeddings)
three_embeddings = np.stack(three_averaged_embeddings)

# Attach embeddings to species dataframes
#five_prime_dataset["embedding"] = list(five_embeddings)
three_prime_dataset["embedding"] = list(three_embeddings)

# Group by species and save each as a separate .npy
# for species, group in five_prime_dataset.groupby("species_proxy"):
#     emb_array = np.stack(group["embedding"].values)
#     species_file = os.path.join(five_out_dir, f"{species}_five_embeddings.npy")
#     print(species_file, "generated")
#     np.save(species_file, emb_array)

for species, group in three_prime_dataset.groupby("species_proxy"):
    emb_array = np.stack(group["embedding"].values)
    species_file = os.path.join(three_out_dir, f"{species}_three_embeddings.npy")
    print(species_file, "generated")
    np.save(species_file, emb_array)

