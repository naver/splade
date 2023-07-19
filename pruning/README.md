# SIGIR23_efficiency

This code shows how to reproduce the pruning experiments from the paper: [A Static Pruning Study on Sparse Neural Retrievers](https://arxiv.org/abs/2304.12702). The following code shows how to take already preprocessed documents to be used with Anserini (generated with splade.create_anserini.py) prune them, index them with Anserini and perform retrieval. To reproduce PISA numbers please follow the [tutorial from JMMackenzie starting from step 2](https://gist.github.com/JMMackenzie/49d7e837751501067cb16d9940d1ad67)

## Step 0: Download data

First thing you need to do is download the data and extract it

```
wget https://www.dropbox.com/s/odkkbgg8lopcduk/pruning.tar.gz?dl=0 -O pruning_data.tar.gz
tar xzvf pruning_data.tar.gz
```

## Step 1: Prune data

To perform all prunings you can just run `bash run_all.sh MODELNAME 1` where modelname is {eff_v_large, eff_v_medium, eff_v_small, msmarco-deepimpact or msmarco-unicoil-tilde}. Note that the script will just create the needed folders and run `bash prune_all.sh MODELNAME`

## Step 2: Indexing with Anserini

Information to come

## Step 3: Retrieving with Anserini

Information to come
