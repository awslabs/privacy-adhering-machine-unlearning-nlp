rm -rf ./containers/
bash example-scripts/mnli/init.sh 5
bash example-scripts/mnli/train.sh 5 2
bash example-scripts/mnli/predict.sh 5
bash example-scripts/mnli/data.sh 5 report-mnli-linear-5-2.csv
bash example-scripts/mnli/memory.sh 5 2
cp -r ./containers /data/sisa/mnli/linear-containers-5-2

rm -rf ./containers/
bash example-scripts/mnli/init.sh 5
bash example-scripts/mnli/train.sh 5 8
bash example-scripts/mnli/predict.sh 5
bash example-scripts/mnli/data.sh 5 report-mnli-linear-5-8.csv
bash example-scripts/mnli/memory.sh 5 8
cp -r ./containers /data/sisa/mnli/linear-containers-5-8

rm -rf ./containers/
bash example-scripts/mnli/init.sh 5
bash example-scripts/mnli/train.sh 5 16
bash example-scripts/mnli/predict.sh 5
bash example-scripts/mnli/data.sh 5 report-mnli-linear-5-16.csv
bash example-scripts/mnli/memory.sh 5 16
cp -r ./containers /data/sisa/mnli/linear-containers-5-16

rm -rf ./containers/
bash example-scripts/mnli/init.sh 5
bash example-scripts/mnli/train.sh 5 32
bash example-scripts/mnli/predict.sh 5
bash example-scripts/mnli/data.sh 5 report-mnli-linear-5-32.csv
bash example-scripts/mnli/memory.sh 5 32
cp -r ./containers /data/sisa/mnli/linear-containers-5-32

cp -r ./results /data/sisa/mnli/results-linear