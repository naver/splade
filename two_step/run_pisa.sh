export dataset=$1
export index_size=$2 #full or pruned
export query_size=$3 #full or pruned

mkdir -p runs_pisa/$dataset
mkdir -p results_pisa/$dataset/
mkdir -p latencies_pisa/$dataset/

./two_step_pisa/build/bin/evaluate_queries \
                                   --encoding block_simdbp \
                                   --documents pisa-canonical/$dataset/$index_size/$index_size.docmap \
                                   --index pisa-index/$dataset/$index_size/block_simdbp.idx \
                                   --wand pisa-index/$dataset/$index_size/quantized-40.bmw \
                                   --algorithm maxscore \
                                   -k 100 \
                                   --scorer quantized \
                                   --weighted \
                                   --queries pisa-index/$dataset/$index_size/${query_size}_queries.pisa.ints \
                                   --run "bp-spladev2" > runs_pisa/$dataset/$index_size-$query_size.trec

python filter_lines.py runs_pisa/$dataset/$index_size-$query_size.trec runs_pisa/$dataset/$index_size-$query_size.fix.trec
ir_measures qrels/qrels.beir-v1.0.0-$dataset.test.txt runs_pisa/$dataset/$index_size-$query_size.fix.trec nDCG@10 > results_pisa/$dataset/$index_size-$query_size.trec
cat results_pisa/$dataset/$index_size-$query_size.trec

./two_step_pisa/build/bin/queries \
                            --encoding block_simdbp \
                            --index pisa-index/$dataset/$index_size/block_simdbp.idx \
                            --wand pisa-index/$dataset/$index_size/quantized-40.bmw \
                            --algorithm maxscore \
                            -k 100 \
                            --weighted \
                            --scorer quantized \
                            --queries pisa-index/$dataset/$index_size/${query_size}_queries.pisa.ints > latencies_pisa/$dataset/$index_size-$query_size.trec

cat latencies_pisa/$dataset/$index_size-$query_size.trec                            
