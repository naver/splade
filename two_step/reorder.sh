export dataset=$1
python reorder.py $dataset
two_step_pisa/build/bin/reorder-docids \
    --collection pisa-canonical/$dataset/full/full \
    --output pisa-canonical/$dataset/full/reorder \
    --from-mapping pisa-canonical/$dataset/reorder

./two_step_pisa/build/bin/compress_inverted_index --encoding block_simdbp --collection pisa-canonical/$dataset/full/reorder --output pisa-index/$dataset/full/reorder.block_simdbp.idx
