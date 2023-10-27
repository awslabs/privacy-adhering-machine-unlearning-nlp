#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

shards=$1
fileName=$2

if [[ ! -f general-report.csv ]]; then
    echo "nb_shards,nb_requests,accuracy,retraining_time" > ./results/${fileName}
fi

for j in {0..15}; do
    echo $j
    r=$((${j}*${shards}/5))
    python aggregation.py --strategy uniform --container "${shards}" --shards "${shards}" --dataset datasets/cola/datasetfile --label "${r}"
    acc=$(python aggregation.py --strategy uniform --container "${shards}" --shards "${shards}" --dataset datasets/cola/datasetfile --label "${r}")
    cat containers/"${shards}"/times/shard-*\:"${r}".time > "containers/${shards}/times/times.tmp" || true
    time=$(python time.py --container "${shards}" | awk -F ',' '{print $1}')
    echo "${shards},${r},${acc},${time}" >> ./results/${fileName}
done
