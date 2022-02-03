# Evaluating SPLADE on MS MARCO with Anserini

In this folder we make available the code for evaluating SPLADE with Anserini. This is an extension of the [Anserini documentation](https://github.com/castorini/anserini/blob/master/docs/experiments-msmarco-passage-splade-v2.md), with the addition of SPLADE inference, where the aforementioned documentation uses precomputed files. In the next paragraphs we will see:

0. (OPTIONAL) Setup 
1. Generating the document collection (jsonl)
2. Generating the topics (query) files
3. Indexing with Anserini
4. How to query the index (and measure latency/QPS)
5. Evaluating the results with trec_eval

Also, you can check `evaluate_splade.sh` for a script that performs most of these steps (safe for installing the environment and anserini). Finally, please notice that there are various licenses here (Anserini, MS MARCO, SPLADE...), so take this into consideration when using the code (notably that it should not be used in commercial applications).

## (OPTIONAL) SETUP

### Environment

There are many requirements for running both SPLADE and Anserini. We suggest the use of the conda environment described by `env.yml`, with a default name of `splade`, which can be installed by:

```
conda env create -f env.yml
```

It can then be activated by:

```
conda activate splade
```

### Anserini

The first thing one needs to do is to install Anserini. [Please follow their guidelines](https://github.com/castorini/anserini). In the following we consider that the path to Anserini is set to `$PATH_ANSERINI`.

### Download MS MARCO needed files

The next step is to download all files related to MS MARCO (documents, development queries and development qrels). This can be done with the following commands:

```
PATH_MSMARCO=msmarco/
URL_MSMARCO=https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz
curl -s $URL_MSMARCO | tar xvz -C $PATH_MSMARCO
```

Note that MS MARCO is a **Non-Commercial** dataset, so that networks trained with it (such as most SPLADE model), **cannot** be used in commercial applications.

### Choose SPLADE weights

The next step is to choose which SPLADE weights you want to use. In this case you can either train it yourself, or download some of the models available below:


| Model                                             |        MS MARCO |      | TREC-2019 | BEIR   NDCG@10 |             | FLOPS | Index size (Gb) | Anserini+Pytorch |
|---------------------------------------------------|-----------------|------|-----------|----------------|-------------|-------|-----------------|------------------|
|                                                   | MRR@10          | R@1k | NDCG@10   | 13 datasets    | 11 datasets |       |                 | QPS on CPU       |
| Baselines                                         |                 |      |           |                |             |       |                 |                  |
| BM25                                              | 18.8            | 85.3 | 50.6      | 43.7           | 47.6        | ???   | **0.5**             | **480**              |
| [ColbertV2 (Santhanam et al   2021)]()            | **39.7**            | **98.4** | ???       | 49.7           | 52.5        | ???   | 20              | ???              |
| SPLADE   v2 paper                                 |                 |      |           |                |             |       |                 |                  |
| [distilSplade_max](https://github.com/naver/splade/tree/main/weights/distilsplade_max)                                  | 36.8            | 97.9 | 72.9      | 49.9           | 52.7        | 3.82  | 4.8             | 22               |
| SPLADE-max (train_max.py)                         |                 |      |           |                |             |       |                 |                  |
| [splade_max_distilbert](http://download-de.europe.naverlabs.com/Splade_Release_Jan22/splade_max_distilbert.tar.gz)                             | 36.8            | 97.7 | 72.4      | 48.7           | 51.5        | 1.14  | 3.2             | 48               |
| [splade_max_CoCodenser](http://download-de.europe.naverlabs.com/Splade_Release_Jan22/splade_distil_CoCodenser_large.tar.gz)                             | 38.2            | **98.4** | 73.6      | 50.2           | **53.1**        | 1.48  | 3.1             | 30               |
| DistilSPLADE-max (train_distill.py)               |                 |      |           |                |             |       |                 |                  |
| [splade_distil_CoCodenser_large](http://download-de.europe.naverlabs.com/Splade_Release_Jan22/splade_distil_CoCodenser_large.tar.gz)                    | 39.3            | 98.3 | 72.5      | 50.1           | 52.8        | 5.35  | 5.9             | 17               |
| [splade_distil_CoCodenser_medium](http://download-de.europe.naverlabs.com/Splade_Release_Jan22/splade_distil_CoCodenser_medium.tar.gz)                   | 38.8            | 98.2 | **74.3**      | **50.3**           | **53.1**        | 1.96  | 3.2             | 29               |
| [splade_distil_CoCodenser_small](http://download-de.europe.naverlabs.com/Splade_Release_Jan22/splade_distil_CoCodenser_small.tar.gz)                    | 37.5            | 97.5 | 71.0      | 46.4           | 48.9        | **0.42**  | 2.0             | 83               |


**Note on latency:**: QPS values are computed on a Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz, with 128 threads. We simply take the total time (lets call it `t`) for query generation (step 2) on that CPU + the total time for Anserini retrieval. We then compute QPS= |Q|/t, where |Q| is the number of queries in the MS MARCO devset (6980).

**Note on BEIR evaluation**: The BEIR dataset is composed of 18 datasets, of which 4 are not directly accessible (BioASQ, Signal1M, TREC-NEWS, Robust04), and 1 has a multiple-faceted evaluation (CQAdupstack). Removing these five datasets, leads us to the "13 datasets" category; some papers also remove NQ and scidocs, leading to the "11 datasets" category.

The models not included in the repo can be downloaded and extracted as follows (example splade_distil_CoCodenser_small):

```
SPLADE_URL=http://download-de.europe.naverlabs.com/Splade_Release_Jan22/splade_distil_CoCodenser_small.tar.gz
SPLADE_NAME=splade_distil_CoCodenser_small
PATH_SPLADE_WEIGHTS=weights/
curl -s $SPLADE_URL | tar xvz -C $PATH_SPLADE_WEIGHTS
```

We then consider that the path to the SPLADE weights is set to the variable $PATH_SPLADE, continuing the example:

```
PATH_SPLADE=$PATH_SPLADE_WEIGHTS/$SPLADE_NAME
```

## 1. Generating the document (jsonl) collection

Now that we have everything setup we may start. Considering everything respects the setup, you may generate the document collection via:

```
SPLIT=16
INDEX_BATCH_SIZE=100
OUTPUT_COLLECTION_PATH=anserini_collection/$SPLADE_NAME
python create_anserini_collection.py --splade_weights_path $PATH_SPLADE --input_collection_path $PATH_MSMARCO/collection.tsv --output_collection_path $OUTPUT_COLLECTION_PATH --index_batch_size $INDEX_BATCH_SIZE --split $SPLIT
```

Where $SPLIT is the number of splits for the collection jsonl (to allow for multi-threaded indexing), $INDEX_BATCH_SIZE is the batch_size for indexing and $OUTPUT_COLLECTION_PATH. It takes around 16h on a T4 GPU to encode the entire MS MARCO collection.

## 2. Generating the topics (query) files

```
TOPIC_BATCH_SIZE=100
TOPIC_NAME=devset_msmarco
OUTPUT_TOPIC_PATH=topics/$SPLADE_NAME
python create_anserini_topic.py --splade_weights_path $PATH_SPLADE --input_topic_path $PATH_MSMARCO/queries.dev.small.tsv --output_topic_path $OUTPUT_TOPIC_PATH --topic_batch_size $TOPIC_BATCH_SIZE --output_topic_name $TOPIC_NAME
```

Where $SPLIT is the number of splits for the collection jsonl (to allow for multi-threaded indexing), $INDEX_BATCH_SIZE is the batch_size for indexing and $OUTPUT_COLLECTION_PATH. If we want to measure latency, we should use TOPIC_BATCH_SIZE=1.

## 3. Indexing with anserini

Now, we just have to index with ANSERINI as follows:

```
INDEX_PATH=anserini_index/$SPLADE_NAME
sh $PATH_ANSERINI/target/appassembler/bin/IndexCollection -collection JsonVectorCollection \
 -input $OUTPUT_COLLECTION_PATH \
 -index $INDEX_PATH \
 -generator DefaultLuceneDocumentGenerator -impact -pretokenized \
 -threads 16
```

## 4. How to query the index (and measure latency/QPS)

With the index created, we can now query the index. We can do it in parallel (recommended) and thus measure QPS (queries per second, parallelism=number of cores) or do it sequentially (threads=1) to measure latency. 

```
RUN_OUTPUT=runs/
sh $PATH_ANSERINI/target/appassembler/bin/SearchCollection -hits 1000 -parallelism 16 \
 -index $INDEX_PATH \
 -topicreader TsvInt -topics $OUTPUT_TOPIC_PATH/$TOPIC_NAME \
 -output $RUN_OUTPUT/$NAME.trec -format trec \
 -impact -pretokenized
```

## 5. Evaluating the results with trec_eval

Finally we can evaluate the results using trec_eval. Here we do MAP, MRR@10 and Recall@k for various values of k. To do so, we first need to convert the qrels from MS MARCO format to trec format:

```
python $PATH_ANSERINI/tools/scripts/msmarco/convert_msmarco_to_trec_qrels.py \
 --input $PATH_MSMARCO/qrels.dev.small.tsv \
 --output $PATH_MSMARCO/qrels.dev.small.trec
```

And now we can run the evaluation

```
$PATH_ANSERINI/tools/eval/trec_eval.9.0.4/trec_eval -c -M 10 -m recip_rank \
$PATH_MSMARCO/qrels.dev.small.trec runs/level_$LEVEL/$NAME.trec

$PATH_ANSERINI/tools/eval/trec_eval.9.0.4/trec_eval -c -mrecall -mmap \
$PATH_MSMARCO/qrels.dev.small.trec runs/level_$LEVEL/$NAME.trec
```
