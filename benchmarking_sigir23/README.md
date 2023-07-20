# CODE/DATA used in Benchmarking Middle-Trained Language Models for Neural Search
## Paper:
 * https://arxiv.org/abs/2306.02867

## Middle-trained models:
* bert-base-uncased
* Shitao/RetroMAE
* Shitao/RetroMAE_MSMARCO
* bert-ms (soon on https://huggingface.co/models?)
* bert-ms-cls (soon on https://huggingface.co/models?)
* LexMAE  (see https://github.com/taoshen58/LexMAE)

## Data
 see https://oss.navercorp.com/nle-sar/learn2index#data


Configurations are under the conf folder.
* config_hf_[dense|splade]_sigir23_Xneg_[no]distil.yaml

To run a config:
```
export PYTHONPATH=$PYTHONPATH:$(pwd)
python -m splade.hf_train init_dict.model_type_or_dir=<YOUR MIDDLETRAINED MODEL>  --config-name <config> config.checkpoint_dir=<PATH>  config.index_dir=<PATHINDEX>  config.out_dir=<PATHRES>  

```
