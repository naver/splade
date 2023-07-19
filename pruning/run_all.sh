export basename=$1
export step=$2

if [[ "$step" == 1 ]]; then

    # Create folders

    mkdir anserini_indexes/$basename
    mkdir anserini_runs/$basename

    # Prune

    bash prune_all.sh $basename

elif [[ "$step" == 2 ]]; then

# When converting to anserini finishes you can use pyserini for indexing

    bash index_all.sh $basename

elif [[ "$step" == 3 ]]; then

# After indexing with pyserini you can then retrieve

    bash query_all_indexes.sh $basename
fi