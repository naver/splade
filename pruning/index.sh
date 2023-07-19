export folder=$1
export name=$2
export base_name=$3

python -m pyserini.index.lucene \
  --collection JsonVectorCollection \
  --input $1 \
  --index anserini_indexes/$base_name/$name \
  --generator DefaultLuceneDocumentGenerator \
  --threads 12 \
  --impact --pretokenized