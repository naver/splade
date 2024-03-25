export dataset=$1
cd beir/$dataset/docs
split docs_anserini.jsonl
rm docs_anserini.jsonl
gzip *
cd ../../../