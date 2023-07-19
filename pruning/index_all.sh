for folder in data/$1/*
do
    f="$(basename -- $folder)"

    echo $folder
    echo $f
    bash index.sh $folder $f $1
done