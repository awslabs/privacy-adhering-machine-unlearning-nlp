cd ./datasets/sst/
python preprocess_data.py 6000
python prepare_data.py
python update_dataset_file.py 4800 1200
cd ..
cd ..
rm -rf ./containers/
bash example-scripts/sst/init.sh 5
bash example-scripts/sst/train.sh 5 16
bash example-scripts/sst/predict.sh 5
bash example-scripts/sst/data.sh 5 report-sst-subsampled-5-16-10.csv
bash example-scripts/sst/memory.sh 5 16
cp -r ./containers /data/sisa/sst/containers-subsampled-5-16-10

cd ./datasets/sst/
python preprocess_data.py 12000
python prepare_data.py
python update_dataset_file.py 9600 2400
cd ..
cd ..
rm -rf ./containers/
bash example-scripts/sst/init.sh 5
bash example-scripts/sst/train.sh 5 16
bash example-scripts/sst/predict.sh 5
bash example-scripts/sst/data.sh 5 report-sst-subsampled-5-16-20.csv
bash example-scripts/sst/memory.sh 5 16
cp -r ./containers /data/sisa/sst/containers-subsampled-5-16-20

cd ./datasets/sst/
python preprocess_data.py 30000
python prepare_data.py
python update_dataset_file.py 24000 6000
cd ..
cd ..
rm -rf ./containers/
bash example-scripts/sst/init.sh 5
bash example-scripts/sst/train.sh 5 16
bash example-scripts/sst/predict.sh 5
bash example-scripts/sst/data.sh 5 report-sst-subsampled-5-16-50.csv
bash example-scripts/sst/memory.sh 5 16
cp -r ./containers /data/sisa/sst/containers-subsampled-5-16-50

cp -r ./results /data/sisa/sst/results-subsampled