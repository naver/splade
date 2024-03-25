mkdir anserini_indexes
export dataset=$1
mkdir anserini_indexes/$dataset/
rm -r anserini_indexes/$dataset/full
java -cp anserini-0.24.1-fatjar.jar io.anserini.index.IndexCollection \
    -collection JsonVectorCollection \
    -input beir/$dataset/docs \
    -index anserini_indexes/$dataset/full \
    -generator DefaultLuceneDocumentGenerator \
    -threads 16 -impact -pretokenized -optimize

rm -r anserini_indexes/$dataset/pruned
java -cp anserini-0.24.1-fatjar.jar io.anserini.index.IndexCollection \
    -collection JsonVectorCollection \
    -input beir/$dataset/docs_pruned \
    -index anserini_indexes/$dataset/pruned \
    -generator DefaultLuceneDocumentGenerator \
    -threads 16 -impact -pretokenized -optimize