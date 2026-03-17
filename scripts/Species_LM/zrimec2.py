import math
import itertools
import collections
import re
import os

import tqdm
import numpy as np
import pandas as pd
import scipy
import glob

import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf

import sklearn
from sklearn import ensemble
from sklearn import pipeline
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, f1_score
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics.pairwise import cosine_similarity
import patsy

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns

#import statannot
import statannotations
from statannotations.Annotator import Annotator
from statannotations.stats.StatTest import StatTest
import plotnine as p9

import Bio.motifs as motifs
# paths 
project_path = "/scratch/prj/lsm_zelezniak/recovered/projects/MSc_project/scripts/specieslm/outputs/"


class Kmerizer:
    
    def __init__(self, k, overlap=True, log=False, divide=False, leave_one_out=False):
        self.k = k
        if leave_one_out:
            self.kmers = {"".join(x):i for i,x in zip(range(4**k), [x for x in itertools.product("ACGT",repeat=k)][:-1])}
        else:
            self.kmers = {"".join(x):i for i,x in zip(range(4**k), itertools.product("ACGT",repeat=k))}
        self.log = log
        self.divide = divide
        self.overlap = overlap
        
    def kmerize(self, seq):
        counts = np.zeros(4**self.k)
        i = 0
        jump = 1 if self.overlap else self.k
        while i < len(seq) - self.k + 1: 
            kmer = seq[i:i+self.k]
            if "N" in kmer:
                i += jump
                continue
            counts[self.kmers[kmer]] += 1
            i += jump
        if self.divide:
            counts = counts/len(seq)
        if self.log:
            counts = np.log(counts + 1)
        return counts
    
    def tokenize(self, seq, jump=False):
        kmers = []
        i = 0
        while i < len(seq) - self.k + 1: 
            kmer = seq[i:i+self.k]
            kmers.append(kmer)
            if jump:
                i += self.k
            else:
                i += 1
        return kmers
    
kmerizer2 = Kmerizer(k=2)
kmerizer3 = Kmerizer(k=3)
kmerizer4 = Kmerizer(k=4)
kmerizer5 = Kmerizer(k=5)
kmerizer6 = Kmerizer(k=6)
kmerizer7 = Kmerizer(k=7)

codonmerizer = Kmerizer(k=3,overlap=False, log=True, divide=True)

def pearson_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    if len(y_pred.shape) == 2:
        y_pred = y_pred.reshape(-1)
    return scipy.stats.pearsonr(y, y_pred)[0]

def pearson_r2(estimator, X, y):
    y_pred = estimator.predict(X)
    #print(estimator[1].alpha_)
    if len(y_pred.shape) == 2:
        y_pred = y_pred.reshape(-1)
    return scipy.stats.pearsonr(y, y_pred)[0] ** 2

def pearson_r2_metric(y_true, y_pred):
    return scipy.stats.pearsonr(y_true, y_pred)[0] ** 2

def spearman_metric(y_true, y_pred):
    return scipy.stats.spearmanr(y_true, y_pred)[0]

import os
def extract_model_and_dataset_from_directory(dir_path):
    dataset_and_model_map = dict()
    for folder in os.listdir(dir_path):
        try:
            _, _, model_type, dataset, _ = folder.split('-')
        except:
            continue
        if not dataset in dataset_and_model_map.keys():
            dataset_and_model_map[dataset] = [folder]
        else:
            dataset_and_model_map[dataset].append(folder)
    return dataset_and_model_map


dataset_and_model_map = extract_model_and_dataset_from_directory('/scratch/prj/lsm_zelezniak/recovered/projects/MSc_project/projects/specieslm/original_data/outputs/')

def get_model_name(path):
    token = " ".join(path.split("-")[2].split("_")[1:])
    model_name  = path.split("-")[1]
    yeast = "yeast only, " if "yeast" in path else "" 
    sacc = "sacc only, " if "_sacc_incl" in path else ""
    token = f"Aware:{token}, " if not 'agnostic' in token else f"Agnostic, "
    sacc_incl = "including all saccs" if "incl_sacc" in path else ""
    sacc_incl = "" if "sacc only," in sacc else sacc_incl
    print (f"{token}{yeast}{sacc}{sacc_incl}".rstrip(', '))
    return (f"{token}{yeast}{sacc}{sacc_incl}".rstrip(', '))

import pprint
pp = pprint.PrettyPrinter(depth=4)
pp.pprint(dataset_and_model_map)

def get_model_name(path):
    token = " ".join(path.split("-")[2].split("_")[1:])
    model_name  = path.split("-")[1]
    yeast = "yeast only, " if "yeast" in path else "" 
    sacc = "sacc only, " if "_sacc_incl" in path else ""
    token = f"Aware:{token}, " if not 'agnostic' in token else f"Agnostic, "
    print (f"{token}".rstrip(', '))
    return (f"{token}".rstrip(', '))

def segal_mpra(path_list):
    mpra_df = pd.read_csv("/scratch/prj/lsm_zelezniak/recovered/projects/MSc_project/projects/specieslm/original_data/data/Downstream_Targets/segal_2015.tsv",sep="\t").dropna().reset_index(drop=True).reset_index()

    mpra_df['TGTAAATA'] = mpra_df['Oligo Sequence'].str.count('TGTAAATA')
    mpra_df['TGCAT'] = mpra_df['Oligo Sequence'].str.count('TGCAT')
    mpra_df['ATATTC'] = mpra_df['Oligo Sequence'].str.count('ATATTC')
    mpra_df['TTTTTTA'] = mpra_df['Oligo Sequence'].str.count('TTTTTTA')
    mpra_df['GC_content_UTR3'] = (mpra_df['Oligo Sequence'].str.count('G') + mpra_df['Oligo Sequence'].str.count('C'))/mpra_df['Oligo Sequence'].str.len()

    def replacer(x):
        x = (x.replace("native_","")
                .replace("pad_","")
                .replace("Lib_pre_","")
                .replace("term_full_","")
                .replace("term_null_","")
                .replace("term_ins_","")
                .replace("lib_suff_nag_","")
                .replace("ib_suff_osz_","")
                .replace("stop_codon_","")
                .replace("canon_term_neg_","")
                .replace("canon_term_pos_","")
                .replace("_no_ins","")
                .replace("_three_ins","")
                .replace("_four_ins","")
                .replace("_two_ins","")
                .replace("_one_ins","")
                .replace("_one","")
                .replace("_synth_insert_pad","")
                .replace("_synth_t","")
                .replace("_512_mut","")
                .replace("_wt","")
               )
        if x.startswith('l'):
            return x[1:]
        else:
            return x

    mpra_df['block'] = [replacer(x.split("#")[1].split("context_")[-1]) for x in mpra_df["Description"]]

    y_obs = np.array(np.log2(mpra_df["Expression"]))
    groups = np.array(mpra_df['block'])

    X_2mer = np.stack(mpra_df['Oligo Sequence'].apply(lambda x: kmerizer2.kmerize(x)))
    X_3mer = np.stack(mpra_df['Oligo Sequence'].apply(lambda x: kmerizer3.kmerize(x)))
    X_4mer = np.stack(mpra_df['Oligo Sequence'].apply(lambda x: kmerizer4.kmerize(x)))
    X_5mer = np.stack(mpra_df['Oligo Sequence'].apply(lambda x: kmerizer5.kmerize(x)))
    X_6mer = np.stack(mpra_df['Oligo Sequence'].apply(lambda x: kmerizer6.kmerize(x)))
    
    segal_emb_dict = {} 
    for path in path_list:
        tmp = np.load(f"{project_path}/{path}/embeddings.npy")            
        segal_emb_dict[get_model_name(path)] = np.array([tmp[i] for i in range(tmp.shape[0])])  
    
    data_matrices = {"2-mer counts":X_2mer, 
                     "3-mer counts":X_3mer, 
                     "4-mer counts":X_4mer,
                     "5-mer counts":X_5mer,
                    }
    data_matrices.update(segal_emb_dict)
    rows = []
    df_list = []
    np.random.seed(42)
    test_predictions = []

    group_kfold = sklearn.model_selection.GroupKFold(n_splits=10)

    for key in data_matrices:
        print(key)
        y = y_obs
        X = data_matrices[key]

        y_pred_list = []
        y_true_list = []
        for i, (train_index, test_index) in enumerate(group_kfold.split(X, y, groups)):
            pipe = pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), sklearn.linear_model.RidgeCV(cv=3, alphas=[10, 100, 1000])) #limit runtime
            X_train = X[train_index]
            y_train = y[train_index]
            X_test = X[test_index]
            y_test = y[test_index]
            pipe.fit(X_train,y_train)
            y_pred = pipe.predict(X_test)
            r2_test = pearson_r2_metric(y_test, y_pred)
            # save results
            y_pred_list.append(y_pred)
            y_true_list.append(y_test)
            rows.append({"model":key, "r2":r2_test})
        test_predictions.append(pd.DataFrame({"model":key, 
                                 "pred": np.concatenate(y_pred_list), 
                                 "true": np.concatenate(y_true_list)}))


###  

#ZRIMEC

###

#Defines the file paths for the upstream and downstream sequences (specieslm embeddings)

downstream_path_list = dataset_and_model_map["scer_downstream_fixedlen"]
upstream_path_list  = dataset_and_model_map["scer_upstream"]
#tpm_values

expr_df_dir = "/scratch/prj/lsm_zelezniak/recovered/projects/MSc_project/results/output_files_species_fungiexp/bc_tpm"

tpm_files = glob.glob(os.path.join(expr_df_dir, "*.tsv"))

expr_df_list = [pd.read_csv(f, sep='\t') for f in tpm_files]


expr_df_list = []
for tpm_file in tpm_files:
    df = pd.read_csv(tpm_file, sep='\t')
    print(df.columns)
    df = df.query('boxcox_tpm >= 5')
    expr_df_list.append(df)

expr_df = pd.concat(expr_df_list).reset_index(drop=True)

for filename in os.listdir(expr_df_dir):
    if filename.endswith(".tsv"):
        file_path = os.path.join(expr_df_dir, filename)
        expr_df = pd.read_csv(file_path)

        # apply box-cox transform with lambda = 0.22
        df["boxcox_TPM"] = scipy.stats.boxcox(df["TPM"], lmbda=0.22)


        # regress out gene length, normalisation step
        # gene_lengths = np.array(expr_df["gene_len"]).reshape(-1, 1)
        # expr_df["residual_boxcox_TPM"] = expr_df["boxcox_TPM"] - sklearn.linear_model.LinearRegression().fit(y=expr_df["boxcox_TPM"], X=gene_lengths).predict(X=gene_lengths)

        # take median over samples
        expr_df = pd.concat(expr_df_list).reset_index(drop=True)
        expr_df = expr_df.groupby('geneId')[["boxcox_tpm","TPM"]].median().reset_index()
        expr_df["log_TPM"] = np.log10(expr_df["TPM"] + 1)

# get test genes
test_genes = pd.read_csv("/scratch/prj/lsm_zelezniak/recovered/projects/MSc_project/projects/specieslm/original_data/data/Downstream_Targets/test_gene_ids.csv")

test_genes_set = set(test_genes["names_test"])
expr_df["is_test"] = expr_df["geneId"].isin(test_genes_set)

five_parquet_files = glob.glob("/scratch/prj/lsm_zelezniak/recovered/projects/MSc_project/originals/Sequences/proj_seq/lab_species/**/*five_prime.parquet", recursive=True)
three_parquet_files = glob.glob("/scratch/prj/lsm_zelezniak/recovered/projects/MSc_project/originals/Sequences/proj_seq/lab_species/**/*three_prime.parquet", recursive=True)

five_sequences = pd.concat([pd.read_parquet(fp) for fp in five_parquet_files], ignore_index=True)
three_sequences = pd.concat([pd.read_parquet(tp) for tp in three_parquet_files], ignore_index=True)

#Creates a dictionary to store the 3' and 5' sequences
three_prime_emb_dict, five_prime_emb_dict = {},{}

#Defining a function to get the model
def get_model_name(path):
    token = " ".join(path.split("-")[2].split("_")[1:])
    if len(re.findall(r'\d+', token)) > 0:
        
        token = " ".join(token.split(" ")[:2])
    model_name  = path.split("-")[1]
    token = f"Aware:{token}, " if not 'agnostic' in token else f"Agnostic, "
    print (f"{token}".rstrip(', '), path)
    return (f"{token}".rstrip(', '))

for path in downstream_path_list:
    tmp = np.load(f"{project_path}/{path}/embeddings.npy")            
    #three_prime_emb_dict["Downstream: " + get_model_name(path)] = [tmp[i] for i in range(tmp.shape[0])]
    three_prime_emb_dict["Downstream: " + model_name] = tmp
    three_sequences["Downstream: " + get_model_name(path)] = [tmp[i] for i in range(tmp.shape[0])]
    three_sequences.drop_duplicates(subset=['geneId'], keep='first', inplace=True)

    
for path in upstream_path_list:
    print (path)
    tmp = np.load(f"{project_path}/{path}/embeddings.npy")            
    #five_prime_emb_dict["Upstream: " + get_model_name(path)] = [tmp[i] for i in range(tmp.shape[0])]
    five_prime_emb_dict["Upstream: " + model_name] = tmp
    five_sequences["Upstream: " + get_model_name(path)] = [tmp[i] for i in range(tmp.shape[0])]

five_sequences.drop_duplicates(subset=['geneId'], keep='first', inplace=True)
#expr_merged = expr_df.merge(five_sequences, on="gene_id").merge(cds_sequences, on="gene_id").merge(three_sequences, on="gene_id")
expr_merged = expr_df.merge(five_sequences, on="geneId").merge(cds_sequences, on="geneId").merge(three_sequences, on="geneId")
y_obs = np.array(expr_merged["boxcox_TPM"])
X_models = {}

for model in three_prime_emb_dict.keys():
    X_models[model] = np.stack(expr_merged[model])

for model in five_prime_emb_dict.keys():
    print (model)
    five_prime_model = np.stack(expr_merged[model])
    X_models[model] = five_prime_model
    if "Aware" in model:
        print (model)
        X_models[f"Combined: Downstream: Aware:candida glabrata ! {model}"] = np.concatenate([five_prime_model, np.stack(expr_merged["Downstream: Aware:candida glabrata"])],axis=1)
    else:
        X_models[f"Combined: Downstream: Agnostic ! {model}"] = np.concatenate([five_prime_model, np.stack(expr_merged["Downstream: Agnostic"])],axis=1)
    #for model_three in three_prime_emb_dict.keys():
    #    X_models[f"Combined: {model_three} ! {model}"] = np.concatenate([X_base, five_prime_model, np.stack(expr_merged[model_three])],axis=1)


#X_base = np.zeros((y_obs.shape[0],1))
X_3mer_five = np.stack(expr_merged['five_prime_seq'].apply(lambda x: kmerizer3.kmerize(x)))
X_4mer_five = np.stack(expr_merged['five_prime_seq'].apply(lambda x: kmerizer4.kmerize(x)))
X_5mer_five = np.stack(expr_merged['five_prime_seq'].apply(lambda x: kmerizer5.kmerize(x)))

X_3mer_three = np.stack(expr_merged['three_prime_seq'].apply(lambda x: kmerizer3.kmerize(x)))
X_4mer_three = np.stack(expr_merged['three_prime_seq'].apply(lambda x: kmerizer4.kmerize(x)))
X_5mer_three = np.stack(expr_merged['three_prime_seq'].apply(lambda x: kmerizer5.kmerize(x)))

X_3mer_combined = np.concatenate([X_3mer_five, X_3mer_three],axis=1)
X_4mer_combined = np.concatenate([X_4mer_five, X_4mer_three],axis=1)
X_5mer_combined = np.concatenate([X_5mer_five, X_5mer_three],axis=1)

data_matrices = {
                 "3-mer-fiveprime":X_3mer_five, 
                 "4-mer-fiveprime":X_4mer_five,
                 "5-mer-fiveprime":X_5mer_five, 
                 "3-mer-threeprime":X_3mer_three,
                 "4-mer-threeprime":X_4mer_three,
                 "5-mer-threeprime":X_5mer_three,
                 "3-mer-combined":X_3mer_combined,
                 "4-mer-combined":X_4mer_combined,
                 "5-mer-combined":X_5mer_combined,
}
data_matrices.update(X_models)

rows = []
df_list = []
total_len = y_obs.shape[0]
test_indices = np.array(expr_merged["is_test"])#.astype(int)
np.random.seed(42)

trained_models = {}

print("Data matrices keys:", list(data_matrices.keys()))
print("Trained models keys:", list(trained_models.keys()))

### REGRESSION MODEL ###


#This loop iterates over each feature key in the data_matrices dictionary
for key in data_matrices:  

    print(key) #Prints the model name
    y = y_obs 
    X = data_matrices[key]

    #Builds a machine learning pipeline
    #StandardScaler() = scales the features
    #RidgeCV()= ridge regresssion model with cross-validation 
    pipe = pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), sklearn.linear_model.RidgeCV(cv=5,alphas=[10,100,1000])) 
   

    pipe.fit(X[~test_indices],y[~test_indices]) #Trains the model
    trained_models[key] = pipe #Store the trained model
   
    y_pred = pipe.predict(X[test_indices]) #Makes predictions on the test set
    r2_val = pearson_r2_metric(y_obs[test_indices], y_pred) #Evaluates the model performance using pearsons r2
    rows.append({"model":key, "r2":r2_val}) #Appends a dictionary with the model name and its R2 score to the rows list
    df_list.append(pd.DataFrame({"model":key,"y_pred":y_pred,"y_true":y_obs[test_indices]}))
        
metrics_expr = pd.DataFrame(rows) #Dataframe of the R2 scores
preds_expr = pd.concat(df_list)

metrics_expr.to_csv("/scratch/prj/lsm_zelezniak/recovered/projects/MSc_project/scripts/specieslm/results/metrics_expr.csv", index=False)
preds_expr.to_csv("/scratch/prj/lsm_zelezniak/recovered/projects/MSc_project/scripts/specieslm/results/preds_expr.csv", index=False)

print(metrics_expr)


tmp_data_matrices= dict()
for key in data_matrices.keys():
    tmp_data_matrices[key] = (data_matrices[key],trained_models[key][1].alpha_)
data_matrices = tmp_data_matrices

#The bootstrap

rows = []
df_list = []
total_len = y_obs.shape[0]
test_indices = np.array(expr_merged["is_test"])     #.astype(int)
np.random.seed(42)

train_indices = np.arange(len(test_indices))[~test_indices]
bootstrapped_models = collections.defaultdict(list)

for key in data_matrices:
    print(key)
    if not "Downstream" in key or "6-mer" in key:
        continue
    y = y_obs
    X, alpha = data_matrices[key]
    for i in tqdm.tqdm(range(100)):
        smpl_indices = np.random.choice(train_indices, size=len(train_indices))
        pipe = pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), sklearn.linear_model.Ridge(alpha=alpha))
        pipe.fit(X[smpl_indices],y[smpl_indices])
        y_pred = pipe.predict(X[test_indices])
        r2_val = pearson_r2_metric(y_obs[test_indices], y_pred)
        rows.append({"model":key, "r2":r2_val})
        bootstrapped_models[key].append(pipe)

metrics_expr_bootstrap = pd.DataFrame(rows)

metrics_expr_bootstrap.groupby('model').std() * 2

metrics_expr_bootstrap.groupby('model').std().sort_values("r2") * 2
