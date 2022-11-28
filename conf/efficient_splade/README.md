Config files for training efficient splade models. 

For training using these configs you have to first download the middle trained PLMs on the root of the repository

```
wget https://www.dropbox.com/s/hir60b9yj194dv7/mlm_flops.tar.gz?dl=0
tar -xzvf mlm_flops.tar.gz?dl=0
```

## Example

To reproduce V large you can use the following command:

```
conda activate splade_env
export PYTHONPATH=$PYTHONPATH:$(pwd)
export SPLADE_CONFIG_NAME="efficient_splade/config_V_large.yaml"
python3 -m splade.all \
  config.checkpoint_dir=experiments/debug/checkpoint \
  config.index_dir=experiments/debug/index \
  config.out_dir=experiments/debug/out
```

Note that we trained with 4 GPUs, so values may differ from the official if trained in a different setting
