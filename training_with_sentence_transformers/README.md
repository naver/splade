# Training SPLADE on MSMARCO v1

Minimalistic code for the training of DistilSPLADE_max models using the [sentence-bert framework](https://github.com/UKPLab/sentence-transformers/). We provide a file for training Splade_max(train_max.py) and one for training distilSplade(train_distill.py). Distillation is done using the latest training data released by Nils Reimer (https://twitter.com/Nils_Reimers/status/1435544757388857345/photo/1). Note that this is not exactly the code used in our papers, and that we are not yet able to provide indexing/retrieve code (but one could maybe use the beir-eval for that), however results are very competitive with state of the art and with what we observed with splade before (MRR, R@1k and NDCG numbers are multiplied by 100 for simplicity):

|                                                              | MSMARCO |       | TREC-2019 |       | TREC-2020 |       |BEIR     |       |
|-------------------------------------------------------------:|:-------:|:-----:|:---------:|:-----:|:---------:|:-----:|:-------:|:-----:|
|                                                        Model |  MRR@10 |  R@1k |           |  R@1k |           |  R@1k | NDCG@10 |FLOPS  |
|                                                    Baselines |         |       |           |       |           |       |         |       |
| distilbert (sentence-transformers/msmarco-distilbert-dot-v5) |   37.3  |  ???  |    70.1   |  ???  |    71.1   |  ???  |         |       |
|        bert (sentence-transformers/msmarco-bert-base-dot-v5) |   38.1  |  ???  |    71.9   |  ???  |    72.3   |  ???  |         |       |
|        Splade_max v2                                         |   34.0  |  96.5 |    68.4   |  85.1 |    ???    |  ???  | 46.4    | 1.32  |
|        distilSplade v2                                       |   36.8  |  97.9 |    72.9   |  86.5 |    71.0   |  83.4 | 50.6    | 3.82  |
|                              Splade_max (train_max.py)       |         |       |           |       |           |       |         |       |
|          distilbert-base-uncased,λq=0.008, λd=0.006          |   35.4  |  96.9 |    69.3               |  80.3     |    67.8   |  77.1 |         | 0.32  |
|          distilbert-base-uncased,λq=0.0008, λd=0.0006        |   36.8  |  97.7 |    72.4               |  82.7     |    70.6   |  78.1 |         | 1.14  |
|          distilbert-base-uncased,λq=0.00008, λd=0.00006      |   36.8  |  98.0 |    72.4               |  **84.7** |    72.0   |  **79.1** | 49.1    | 3.39  |
|               Luyu/co-condenser-wiki,λq=0.0008, λd=0.0006    |   37.2  |  98.0 |    69.6               |  83.1     |    **72.8**   |  79.0 |         | 1.26  |
|              Luyu/co-condenser-marco,λq=0.0008, λd=0.0006    |   **38.2**  |  **98.5** |    **73.6**   |  84.3     |  72.4     |  78.7 |         | 1.48  |
|                              DistilSplade (train_distill.py) |         |       |           |       |           |       |         |       |
|          distilbert-base-uncased,λq=0.1, λd=0.08             |   38.2  |  97.8 |    73.8   |  87.0 |    71.5   |  82.6 |         | 1.95  |
|          distilbert-base-uncased,λq=0.01, λd=0.008           |   38.5  |  98.0 |    **74.2**   |  **87.8** |    71.9   |  82.6 | 50.1    | 3.85  |
|          distilbert-base-uncased,λq=0.001, λd=0.0008         |   38.7  |  98.1 |    72.4   |  87.0 |    71.7   |  83.4 |         | 7.81  |
|               Luyu/co-condenser-wiki,λq=0.01, λd=0.008       |   38.7  |  98.2 |    73.3   |  87.0 |    72.4   |  83.0 |         | 4.57  |
|              Luyu/co-condenser-marco,λq=0.01, λd=0.008       |   **39.3**  |  **98.4** |    72.5   |  **87.8** |    **73.0**   |  **83.5** | **51.0**    | 5.35  |


## Differences from paper:

There are some differences in this training code to the one we used in the SpladeV2 paper namely:

* For SpladeMax: 
    * Instead of BM25 negatives we use the ones from: https://twitter.com/Nils_Reimers/status/1435544757388857345/photo/1
    * We also use their strategy to remove hard negatives that maybe are not negatives at all (depending on the score of a cross attention model)
    * Training hyperparameters (epoch, lr, warmup...)
* For DistilSplade
    * Instead of DistilSplade negatives we use the ones from: https://twitter.com/Nils_Reimers/status/1435544757388857345/photo/1
    * Instead of using our cross attention scores, we use their negative scores
    * Training hyperparameters (epoch, lr, warmup...)
* Base networks
    * In the paper we always use distilbert-base-uncased
    * In this experiments the base network is explicitly added
