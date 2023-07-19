for name in $1
do
    for value in 0.5 0.75 0.85
    do 
        bash prune_quantile.sh $name $value
    done
done