#!/bin/bash
# This file performs SPLADE evaluation, considering that the environment is already correct and that Anserini is installed in PATH ANSERINI. 
# Also it will always download MS MARCO files and SPLADE weights, feel free to comment these lines if not needed.

### Definitions

### ANSERINI (NEED TO SET PATH)
export PATH_ANSERINI=SETPATH

### MS MARCO
export URL_MSMARCO=https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz
export PATH_MSMARCO=msmarco/

### SPLADE
export SPLADE_URL=http://download-de.europe.naverlabs.com/Splade_Release_Jan22/splade_distil_CoCodenser_large.tar.gz
export PATH_SPLADE_WEIGHTS=weights/
export SPLADE_NAME=splade_distil_CoCodenser_large
export PATH_SPLADE=$PATH_SPLADE_WEIGHTS/$SPLADE_NAME

### COLLECTION
export SPLIT=16
export INDEX_BATCH_SIZE=200
export OUTPUT_COLLECTION_PATH=anserini_collection/$SPLADE_NAME

### TOPIC
export TOPIC_BATCH_SIZE=100
export TOPIC_NAME=devset_msmarco
export OUTPUT_TOPIC_PATH=topics/$SPLADE_NAME

### INDEXING
export INDEX_PATH=anserini_index/$SPLADE_NAME

### QUERYING
export RUN_OUTPUT=runs/

## Download MS MARCO

curl -s $URL_MSMARCO --progress-bar --verbose | tar xvz -C $PATH_MSMARCO

## Download SPLADE weights
curl -s $SPLADE_URL --progress-bar --verbose | tar xvz -C $PATH_SPLADE_WEIGHTS


## 1. Create collection

python create_anserini_collection.py --splade_weights_path $PATH_SPLADE --input_collection_path $PATH_MSMARCO/collection.tsv --output_collection_path $OUTPUT_COLLECTION_PATH --index_batch_size $INDEX_BATCH_SIZE --split $SPLIT

## 2. Create topics

python create_anserini_topic.py --splade_weights_path $PATH_SPLADE --input_topic_path $PATH_MSMARCO/queries.dev.small.tsv --output_topic_path $OUTPUT_TOPIC_PATH --topic_batch_size $TOPIC_BATCH_SIZE --output_topic_name $TOPIC_NAME

## 3. Index with anserini

sh $PATH_ANSERINI/target/appassembler/bin/IndexCollection -collection JsonVectorCollection \
 -input $OUTPUT_COLLECTION_PATH \
 -index $INDEX_PATH \
 -generator DefaultLuceneDocumentGenerator -impact -pretokenized \
 -threads 16

## 4. Query the index

sh $PATH_ANSERINI/target/appassembler/bin/SearchCollection -hits 1000 -parallelism 128 \
 -index $INDEX_PATH \
 -topicreader TsvInt -topics $OUTPUT_TOPIC_PATH/$TOPIC_NAME.tsv \
 -output $RUN_OUTPUT/$NAME.trec -format trec \
 -impact -pretokenized

## 5. Evaluate with trec_eval

python $PATH_ANSERINI/tools/scripts/msmarco/convert_msmarco_to_trec_qrels.py \
 --input $PATH_MSMARCO/qrels.dev.small.tsv \
 --output $PATH_MSMARCO/qrels.dev.small.trec

$PATH_ANSERINI/tools/eval/trec_eval.9.0.4/trec_eval -c -M 10 -m recip_rank \
$PATH_MSMARCO/qrels.dev.small.trec $RUN_OUTPUT/$NAME.trec

$PATH_ANSERINI/tools/eval/trec_eval.9.0.4/trec_eval -c -mrecall -mmap \
$PATH_MSMARCO/qrels.dev.small.trec $RUN_OUTPUT/$NAME.trec
