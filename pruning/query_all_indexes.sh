for folder in anserini_indexes/$1/*
do
    f="$(basename -- $folder)"

    echo $folder
    echo $f
    bash query_index.sh $folder $f $1
done