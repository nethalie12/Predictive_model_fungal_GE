# Developing a Pipeline for Gene Expression Prediction in Unconventional Fungi 
This project aims to predict gene expression from a dataset of 205 fungi using DNA language models and ridge regression.

## Description
Fungi are a valued method of protein production across several industries. Improving rates of fungal protein production is important, however current methods can be time consuming and inefficient. Instead, transcription can be regulated by manipulating fungal DNA regulatory regions to increase mRNA abundance and ultimately protein production. 
Yet, there is limited understanding of the link between DNA regulatory regions and gene expression. Although machine learning techniques can process complex regulatory regions, current research focuses on a limited group of model organisms. Considering the large size of the fungal kingdom, exploring non-model fungi as industrial tools is useful.

This project aims to shift attention towards unconventional fungi as an industrial tool by predicting rates of gene expression across a 205 fungal species. This dataset comprises 205 Transcript Per Million (TPM) gene expression counts sourced from the FungiExp database. SpeciesLM DNA language models were used to process fungal DNA sequences, 
with ridge regression used for gene expression prediction. By examining rates of gene expression across various fungi, the link between regulatory regions and gene expression can be explored, to refine industrial methods of protein production. 

## Getting Started

### Dataset

Fungal TPM values sourced from FungiExp database for fungal gene expression and alternative splicing database. 
https://academic.oup.com/bioinformatics/article/39/1/btad042/6992664


Species DNA sequences sourced from Ensembl Fungi and the National Center for Biotechnology Information (NCBI) databases.
NCBI: https://www.ncbi.nlm.nih.gov/
EnsemblFungi: http://fungi.ensembl.org/index.html

### Models

Species aware DNA language models sourced from HuggingFace.
https://huggingface.co/gagneurlab/SpeciesLM
