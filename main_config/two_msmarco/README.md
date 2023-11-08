# SIGIR23_Two_MSMARCO

This code shows how to reproduce the SPLADE experiments in Table 1 from the paper: [The Tale of two MSMARCO - and their unfair comparisons](https://arxiv.org/pdf/2304.12904.pdf). 

## Step 0: Download data

First thing you need to do is download the data and extract it

```
wget https://www.dropbox.com/s/te0qblvvbba76q3/collection_with_titles.tar.gz?dl=0 -O collection_with_titles.tar.gz
tar xzvf collection_with_titles.tar.gz
```

## Step 1: Train

For training the splade without titles:

```
python -m torch.distributed.launch --use_env --nproc_per_node NUMGPU  -m splade.hf_train  --config-name=splade_default --config-dir=main_config/two_msmarco
```

For training the splade with titles:

```
python -m torch.distributed.launch --use_env --nproc_per_node NUMGPU  -m splade.hf_train  --config-name=splade_titles --config-dir=main_config/two_msmarco
```

## Step 2: Indexing 

For indexing the splade trained without titles:

```
python -m splade.index --config-name=splade_default --config-dir=main_config/two_msmarco
```

And the one with titles

```
python -m splade.index --config-name=splade_titles --config-dir=main_config/two_msmarco
```

## Step 3: Retrieving and evaluating with Numba

For indexing the splade trained without titles:

```
python -m splade.retrieve --config-name=splade_default --config-dir=main_config/two_msmarco
```

And the one with titles

```
python -m splade.retrieve --config-name=splade_titles --config-dir=main_config/two_msmarco
```

