#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

shards=$1
slices=$2

for i in $(seq 0 "$((${shards}-1))"); do
    for j in {0..15}; do
        echo "shard: $((${i}+1))/${shards}, requests: $((${j}+1))/16"
        r=$((${j}*${shards}/5))
        python sisa.py --model bert --train --slices ${slices} --dataset datasets/mnli/datasetfile --label "${r}" --epochs 10 --batch_size 16 --learning_rate 0.001 --optimizer sgd --chkpt_interval 1 --container "${shards}" --shard "${i}"
    done
done
