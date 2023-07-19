for name in $1
do
    for size in 4 8 16 32 64
    do 
        bash prune_doc_index.sh $name 0 $size
    done
done

for name in $1
do
    for value in 0.5 0.75 1.0 1.25 
    do 
        bash prune_doc_index.sh $name $value 0
    done
done