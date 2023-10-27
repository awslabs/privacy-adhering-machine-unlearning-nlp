rm -rf ./containers/
bash example-scripts/cola/init.sh 5
bash example-scripts/cola/train.sh 5 2
bash example-scripts/cola/predict.sh 5
bash example-scripts/cola/data.sh 5 report-cola-5-2.csv
bash example-scripts/cola/memory.sh 5 2

rm -rf ./containers/
bash example-scripts/cola/init.sh 5
bash example-scripts/cola/train.sh 5 20
bash example-scripts/cola/predict.sh 5
bash example-scripts/cola/data.sh 5 report-cola-5-20.csv
bash example-scripts/cola/memory.sh 5 20

rm -rf ./containers/
bash example-scripts/cola/init.sh 5
bash example-scripts/cola/train.sh 5 200
bash example-scripts/cola/predict.sh 5
bash example-scripts/cola/data.sh 5 report-cola-5-200.csv
bash example-scripts/cola/memory.sh 5 200

rm -rf ./containers/
bash example-scripts/cola/init.sh 5
bash example-scripts/cola/train.sh 5 2000
bash example-scripts/cola/predict.sh 5
bash example-scripts/cola/data.sh 5 report-cola-5-2000.csv
bash example-scripts/cola/memory.sh 5 2000