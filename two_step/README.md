# two-step-splade

This is the code for the Two-Step SPLADE paper (https://link.springer.com/chapter/10.1007/978-3-031-56060-6_23). There is two ways of using this code: Either you can replicate our results using our precomputed indexes or you can reproduce it by generating the indexes.

## Replicating Results

1. Download all index and query files from: `wget -O two_step.tar.gz "https://www.dropbox.com/scl/fi/gjl9x8wg08bdmkic0x7xa/two_step.tar.gz?rlkey=x03evbuvnamqml6v68m5scf5x&dl=1"`
2. Untar `tar -zxvf two_step.tar.gz`
3. Install pisa (from our folder)
4. Run the desired table line (method #) for each dataset (example `bash run_method_b.sh $dataset`)
5. Aggregate results

## Reproduce

1. Install all submodules (including our pisa)
2. Download anserini fatjar `wget https://repo1.maven.org/maven2/io/anserini/anserini/0.24.1/anserini-0.24.1-fatjar.jar`
3. Generate anserini files with `bash run_splade.sh $dataset`
4. Split and gzip files with `bash split.sh $dataset`
5. Count doc tokens with `python token_count.py $dataset`
6. Count query tokens with `python token_count_query.py $dataset`
7. Prune `python prune.py $dataset`
8. Index files with `bash index.sh $dataset`
  8. Test anserini indexes `bash retrieve.sh $dataset`
9. Convert to pisa with `bash convert_pisa.sh $dataset`
10. Reorder full index with `bash reorder.sh $dataset`
11. Run the desired table line (method #) for each dataset (example `bash run_method_b.sh $dataset`)
12. Aggregate results

## Issues or problems

Feel free to create a new issue or to send me an email directly (cadurosar@gmail.com)

## Cite

```
@inproceedings{lassance2024two,
  title={Two-Step SPLADE: Simple, Efficient and Effective Approximation of SPLADE},
  author={Lassance, Carlos and Dejean, Herv{\'e} and Clinchant, St{\'e}phane and Tonellotto, Nicola},
  booktitle={European Conference on Information Retrieval},
  pages={349--363},
  year={2024},
  organization={Springer}
}
```
