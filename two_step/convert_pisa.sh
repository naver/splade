export dataset=$1
mkdir pisa-canonical
mkdir ciff_output

./ciff/target/appassembler/bin/ExportAnseriniLuceneIndex -index anserini_indexes/$dataset/full -output ciff_output/$dataset-full.ciff
./ciff/target/appassembler/bin/ExportAnseriniLuceneIndex -index anserini_indexes/$dataset/pruned -output ciff_output/$dataset-pruned.ciff

./faster-graph-bisection/target/release/create-rgb --input ciff_output/$dataset-full.ciff --min-len 128 --output-ciff ciff_output/$dataset-full-bp.ciff
./faster-graph-bisection/target/release/create-rgb --input ciff_output/$dataset-pruned.ciff --min-len 128 --output-ciff ciff_output/$dataset-pruned-bp.ciff
rm create-rgb-*

rm -r pisa-canonical/$dataset/full/
rm -r pisa-canonical/$dataset/pruned/
mkdir -p pisa-canonical/$dataset/full/
mkdir -p pisa-canonical/$dataset/pruned/

./pisa-ciff/target/release/ciff2pisa --ciff-file ciff_output/$dataset-full-bp.ciff --output pisa-canonical/$dataset/full/full
./pisa-ciff/target/release/ciff2pisa --ciff-file ciff_output/$dataset-pruned-bp.ciff --output pisa-canonical/$dataset/pruned/pruned

rm -r pisa-index/$dataset/full
rm -r pisa-index/$dataset/pruned
mkdir -p pisa-index/$dataset/full
mkdir -p pisa-index/$dataset/pruned

# 1. We'll build a compressed index with SIMD-BP128 compression
./two_step_pisa/build/bin/compress_inverted_index --encoding block_simdbp --collection pisa-canonical/$dataset/full/full --output pisa-index/$dataset/full/block_simdbp.idx
./two_step_pisa/build/bin/compress_inverted_index --encoding block_simdbp --collection pisa-canonical/$dataset/pruned/pruned --output pisa-index/$dataset/pruned/block_simdbp.idx

# 2. We'll build the WAND metadata for dynamic pruning. We'll use fixed-sized blocks of 40 elements. Note that since spladev2 comes pre-quantized (the frequencies store impacts) we need to tell PISA to use quantized scoring.
./two_step_pisa/build/bin/create_wand_data --collection pisa-canonical/$dataset/full/full --block-size 40 --scorer quantized --output pisa-index/$dataset/full/quantized-40.bmw
./two_step_pisa/build/bin/create_wand_data --collection pisa-canonical/$dataset/full/full --block-size 40 --scorer saturated_tf --bm25-k1 100 --output pisa-index/$dataset/full/saturated-40.bmw

./two_step_pisa/build/bin/create_wand_data --collection pisa-canonical/$dataset/pruned/pruned --block-size 40 --scorer quantized --output pisa-index/$dataset/pruned/quantized-40.bmw
./two_step_pisa/build/bin/create_wand_data --collection pisa-canonical/$dataset/pruned/pruned --block-size 40 --scorer saturated_tf --bm25-k1 100 --output pisa-index/$dataset/pruned/saturated-40.bmw


# 3. We'll build the document map, and the lexicon map, though we don't need the term lexicon (you will see why soon)
./two_step_pisa/build/bin/lexicon build pisa-canonical/$dataset/full/full.documents pisa-canonical/$dataset/full/full.docmap
./two_step_pisa/build/bin/lexicon build pisa-canonical/$dataset/full/full.terms pisa-canonical/$dataset/full/full.termmap

./two_step_pisa/build/bin/lexicon build pisa-canonical/$dataset/pruned/pruned.documents pisa-canonical/$dataset/pruned/pruned.docmap
./two_step_pisa/build/bin/lexicon build pisa-canonical/$dataset/pruned/pruned.terms pisa-canonical/$dataset/pruned/pruned.termmap

cat beir/$dataset/queries_anserini.tsv | sed -e's/\t/:/' > pisa-index/$dataset/full/full_queries.pisa
sed -e's/:/: /' -i pisa-index/$dataset/full/full_queries.pisa
awk -F" " 'NR==FNR{a[$1]=i++;next}{printf $1; for(i=2; i <= NF; i++){printf " "a[$i]} print ""}' pisa-canonical/$dataset/full/full.terms pisa-index/$dataset/full/full_queries.pisa > pisa-index/$dataset/full/full_queries.pisa.ints

cat beir/$dataset/queries_anserini.tsv | sed -e's/\t/:/' > pisa-index/$dataset/pruned/full_queries.pisa
sed -e's/:/: /' -i pisa-index/$dataset/pruned/full_queries.pisa
awk -F" " 'NR==FNR{a[$1]=i++;next}{printf $1; for(i=2; i <= NF; i++){printf " "a[$i]} print ""}' pisa-canonical/$dataset/pruned/pruned.terms pisa-index/$dataset/pruned/full_queries.pisa > pisa-index/$dataset/pruned/full_queries.pisa.ints

cat beir/$dataset/pruned_anserini.tsv | sed -e's/\t/:/' > pisa-index/$dataset/full/pruned_queries.pisa
sed -e's/:/: /' -i pisa-index/$dataset/full/pruned_queries.pisa
awk -F" " 'NR==FNR{a[$1]=i++;next}{printf $1; for(i=2; i <= NF; i++){printf " "a[$i]} print ""}' pisa-canonical/$dataset/full/full.terms pisa-index/$dataset/full/pruned_queries.pisa > pisa-index/$dataset/full/pruned_queries.pisa.ints

cat beir/$dataset/pruned_anserini.tsv | sed -e's/\t/:/' > pisa-index/$dataset/pruned/pruned_queries.pisa
sed -e's/:/: /' -i pisa-index/$dataset/pruned/pruned_queries.pisa
awk -F" " 'NR==FNR{a[$1]=i++;next}{printf $1; for(i=2; i <= NF; i++){printf " "a[$i]} print ""}' pisa-canonical/$dataset/pruned/pruned.terms pisa-index/$dataset/pruned/pruned_queries.pisa > pisa-index/$dataset/pruned/pruned_queries.pisa.ints
