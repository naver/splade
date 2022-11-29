## Step 0 - Install pisa from the weight-queries branch

The pre-requisite step is to install pisa. Note that it could take a while. 

```
git clone https://github.com/pisa-engine/pisa.git
cd pisa
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
cd ../../ 
```

## Step 1 - Download indexes and queries and extract here

```
wget https://www.dropbox.com/s/odkkbgg8lopcduk/pisa_index.tar.gz?dl=0 -O pisa_index.tar.gz
tar xzvf pisa_index.tar.gz
```

## Step 2 - Run parallel retrieval to get QPS and effectiveness

```
export level=level_5 #level_5 for V) and level_6 for VI)
export size=large # small, medium or large

pisa/build/bin/evaluate_queries \
                                   --encoding block_simdbp \
                                   --documents indexes/$level/$size.docmap \
                                   --index indexes/$level/$size.block_simdbp.idx \
                                   --wand indexes/$level/$size.fixed-40.bmw \
                                   --algorithm block_max_wand \
                                   -k 1000 \
                                   --scorer quantized \
                                   --weighted \
                                   --queries queries/$level/$size.pisa.ints \
                                   --run "$level/$size" > ${level}_${size}.trec

python -m pyserini.eval.trec_eval -c -M 10 -m recip_rank msmarco-passage-dev-subset ${level}_${size}.trec
python -m pyserini.eval.trec_eval -c -mrecall msmarco-passage-dev-subset ${level}_${size}.trec
```

## Step 3 - Run mono-batch mono-cpu retrieval to get average latency


```
level=level_6 #level_5 for V) and level_6 for VI)
size=medium # small, medium or large

pisa/build/bin/queries \
                            --encoding block_simdbp \
                            --index indexes/$level/$size.block_simdbp.idx \
                            --wand indexes/$level/$size.fixed-40.bmw \
                            --algorithm block_max_wand \
                            -k 1000 \
                            --scorer quantized \
                            --weighted \
                            --queries queries/$level/$size.pisa.ints
```
