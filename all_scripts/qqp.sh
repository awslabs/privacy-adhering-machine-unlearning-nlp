rm -rf ./containers/
bash example-scripts/qqp/init.sh 5
bash example-scripts/qqp/train.sh 5 2
bash example-scripts/qqp/predict.sh 5
bash example-scripts/qqp/data.sh 5 report-qqp-linear-5-2.csv
bash example-scripts/qqp/memory.sh 5 2
cp -r ./containers /data/sisa/qqp/linear-containers-5-2

rm -rf ./containers/
bash example-scripts/qqp/init.sh 5
bash example-scripts/qqp/train.sh 5 8
bash example-scripts/qqp/predict.sh 5
bash example-scripts/qqp/data.sh 5 report-qqp-linear-5-8.csv
bash example-scripts/qqp/memory.sh 5 8
cp -r ./containers /data/sisa/qqp/linear-containers-5-8

rm -rf ./containers/
bash example-scripts/qqp/init.sh 5
bash example-scripts/qqp/train.sh 5 16
bash example-scripts/qqp/predict.sh 5
bash example-scripts/qqp/data.sh 5 report-qqp-linear-5-16.csv
bash example-scripts/qqp/memory.sh 5 16
cp -r ./containers /data/sisa/qqp/linear-containers-5-16

rm -rf ./containers/
bash example-scripts/qqp/init.sh 5
bash example-scripts/qqp/train.sh 5 32
bash example-scripts/qqp/predict.sh 5
bash example-scripts/qqp/data.sh 5 report-qqp-linear-5-32.csv
bash example-scripts/qqp/memory.sh 5 32
cp -r ./containers /data/sisa/qqp/linear-containers-5-32

cp -r ./results /data/sisa/qqp/results-linear