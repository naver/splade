#!/bin/bash
#SBATCH --output=logs/msmarco_all_1encoder_adaptertune_houlsby_hp_flops_ld9e5/beir_eval.log
#SBATCH --error=logs/msmarco_all_1encoder_adaptertune_houlsby_hp_flops_ld9e5/beir_eval.err
source ~/.bashrc
conda activate splade

export PYTHONPATH=$PYTHONPATH:$(pwd)
export SPLADE_CONFIG_FULLPATH="/scratch/1/user/vpal/sync_splade/splade/conf/config_splade_msmarco_adapters_regularization_increased2.yaml"

for dataset in nfcorpus arguana fiqa quora scidocs scifact trec-covid webis-touche2020 climate-fever nq fever dbpedia-entity hotpotqa 
do
      sbatch beir.sh \
      +beir.dataset=$dataset \
      +beir.dataset_path=data/beir \
      config.index_retrieve_batch_size=100
done
#  