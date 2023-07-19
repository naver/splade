export folder=$1
export name=$2
export base_name=$3

python -m pyserini.search.lucene \
    --index $folder \
    --topics data/queries/$base_name/MSMARCO.tsv \
    --output anserini_runs/$base_name/$name.txt \
    --output-format trec \
    --batch 100 --threads 16 \
    --hits 1000 \
    --impact

ir_measures msmarco-passage/dev/small anserini_runs/$base_name/$name.txt MRR@10 nDCG@10 R@1000