cd ./datasets/sst/
python preprocess_data.py 60000
python prepare_data.py
python update_dataset_file.py 48000 12000
cd ..
cd ..
rm -rf ./containers/
bash example-scripts/sst/init.sh 5
bash example-scripts/sst/train.sh 5 16
bash example-scripts/sst/predict.sh 5
bash example-scripts/sst/data.sh 5 report-sst-1-5-16.csv
bash example-scripts/sst/memory.sh 5 16
cp -r ./containers /data/sisa/sst/1-containers-5-16

cd ./datasets/sst/
python preprocess_data.py 60000
python prepare_data.py
python update_dataset_file.py 48000 12000
cd ..
cd ..
rm -rf ./containers/
bash example-scripts/sst/init.sh 5
bash example-scripts/sst/train.sh 5 16
bash example-scripts/sst/predict.sh 5
bash example-scripts/sst/data.sh 5 report-sst-2-5-16.csv
bash example-scripts/sst/memory.sh 5 16
cp -r ./containers /data/sisa/sst/2-containers-5-16

cd ./datasets/sst/
python preprocess_data.py 60000
python prepare_data.py
python update_dataset_file.py 48000 12000
cd ..
cd ..
rm -rf ./containers/
bash example-scripts/sst/init.sh 5
bash example-scripts/sst/train.sh 5 16
bash example-scripts/sst/predict.sh 5
bash example-scripts/sst/data.sh 5 report-sst-3-5-16.csv
bash example-scripts/sst/memory.sh 5 16
cp -r ./containers /data/sisa/sst/3-containers-5-16

cd ./datasets/sst/
python preprocess_data.py 60000
python prepare_data.py
python update_dataset_file.py 48000 12000
cd ..
cd ..
rm -rf ./containers/
bash example-scripts/sst/init.sh 5
bash example-scripts/sst/train.sh 5 32
bash example-scripts/sst/predict.sh 5
bash example-scripts/sst/data.sh 5 report-sst-1-5-32.csv
bash example-scripts/sst/memory.sh 5 32
cp -r ./containers /data/sisa/sst/1-containers-5-32

cd ./datasets/sst/
python preprocess_data.py 60000
python prepare_data.py
python update_dataset_file.py 48000 12000
cd ..
cd ..
rm -rf ./containers/
bash example-scripts/sst/init.sh 5
bash example-scripts/sst/train.sh 5 32
bash example-scripts/sst/predict.sh 5
bash example-scripts/sst/data.sh 5 report-sst-2-5-32.csv
bash example-scripts/sst/memory.sh 5 32
cp -r ./containers /data/sisa/sst/2-containers-5-32

cd ./datasets/sst/
python preprocess_data.py 60000
python prepare_data.py
python update_dataset_file.py 48000 12000
cd ..
cd ..
rm -rf ./containers/
bash example-scripts/sst/init.sh 5
bash example-scripts/sst/train.sh 5 32
bash example-scripts/sst/predict.sh 5
bash example-scripts/sst/data.sh 5 report-sst-3-5-32.csv
bash example-scripts/sst/memory.sh 5 32
cp -r ./containers /data/sisa/sst/3-containers-5-32

cp -r ./results /data/sisa1/sst/results-runs