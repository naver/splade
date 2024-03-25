export PYTHONPATH=$PYTHONPATH:$(pwd)
export SPLADE_CONFIG_NAME="config_splade_two_step"

cd ../

dataset=$1 

python3 -m splade.create_anserini_beir \
  +quantization_factor_document=100 \
  +beir.dataset=$dataset \
  +beir.dataset_path=data/beir \
  +beir.split=test \
  +quantization_factor_query=100
