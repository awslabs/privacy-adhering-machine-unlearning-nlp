rm -rf ./containers/
bash example-scripts/rte/init.sh 5
bash example-scripts/rte/train.sh 5 2
bash example-scripts/rte/predict.sh 5
bash example-scripts/rte/data.sh 5 report-rte-5-2.csv
bash example-scripts/rte/memory.sh 5 2

rm -rf ./containers/
bash example-scripts/rte/init.sh 5
bash example-scripts/rte/train.sh 5 20
bash example-scripts/rte/predict.sh 5
bash example-scripts/rte/data.sh 5 report-rte-5-20.csv
bash example-scripts/rte/memory.sh 5 20

rm -rf ./containers/
bash example-scripts/rte/init.sh 5
bash example-scripts/rte/train.sh 5 200
bash example-scripts/rte/predict.sh 5
bash example-scripts/rte/data.sh 5 report-rte-5-200.csv
bash example-scripts/rte/memory.sh 5 200

rm -rf ./containers/
bash example-scripts/rte/init.sh 5
bash example-scripts/rte/train.sh 5 2000
bash example-scripts/rte/predict.sh 5
bash example-scripts/rte/data.sh 5 report-rte-5-2000.csv
bash example-scripts/rte/memory.sh 5 2000