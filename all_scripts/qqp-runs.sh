cd ./datasets/qqp/
python preprocess_data.py 60000
python prepare_data.py
python update_dataset_file.py 48000 12000
cd ..
cd ..
rm -rf ./containers/
bash example-scripts/qqp/init.sh 5
bash example-scripts/qqp/train.sh 5 2
bash example-scripts/qqp/predict.sh 5
bash example-scripts/qqp/data.sh 5 report-qqp-1-5-2.csv
bash example-scripts/qqp/memory.sh 5 2
cp -r ./containers /data/sisa/qqp/1-containers-5-2

cd ./datasets/qqp/
python preprocess_data.py 60000
python prepare_data.py
python update_dataset_file.py 48000 12000
cd ..
cd ..
rm -rf ./containers/
bash example-scripts/qqp/init.sh 5
bash example-scripts/qqp/train.sh 5 2
bash example-scripts/qqp/predict.sh 5
bash example-scripts/qqp/data.sh 5 report-qqp-2-5-2.csv
bash example-scripts/qqp/memory.sh 5 2
cp -r ./containers /data/sisa/qqp/2-containers-5-2

cd ./datasets/qqp/
python preprocess_data.py 60000
python prepare_data.py
python update_dataset_file.py 48000 12000
cd ..
cd ..
rm -rf ./containers/
bash example-scripts/qqp/init.sh 5
bash example-scripts/qqp/train.sh 5 2
bash example-scripts/qqp/predict.sh 5
bash example-scripts/qqp/data.sh 5 report-qqp-3-5-2.csv
bash example-scripts/qqp/memory.sh 5 2
cp -r ./containers /data/sisa/qqp/3-containers-5-2

cd ./datasets/qqp/
python preprocess_data.py 60000
python prepare_data.py
python update_dataset_file.py 48000 12000
cd ..
cd ..
rm -rf ./containers/
bash example-scripts/qqp/init.sh 5
bash example-scripts/qqp/train.sh 5 8
bash example-scripts/qqp/predict.sh 5
bash example-scripts/qqp/data.sh 5 report-qqp-1-5-8.csv
bash example-scripts/qqp/memory.sh 5 8
cp -r ./containers /data/sisa/qqp/1-containers-5-8

cd ./datasets/qqp/
python preprocess_data.py 60000
python prepare_data.py
python update_dataset_file.py 48000 12000
cd ..
cd ..
rm -rf ./containers/
bash example-scripts/qqp/init.sh 5
bash example-scripts/qqp/train.sh 5 8
bash example-scripts/qqp/predict.sh 5
bash example-scripts/qqp/data.sh 5 report-qqp-2-5-8.csv
bash example-scripts/qqp/memory.sh 5 8
cp -r ./containers /data/sisa/qqp/2-containers-5-8

cd ./datasets/qqp/
python preprocess_data.py 60000
python prepare_data.py
python update_dataset_file.py 48000 12000
cd ..
cd ..
rm -rf ./containers/
bash example-scripts/qqp/init.sh 5
bash example-scripts/qqp/train.sh 5 8
bash example-scripts/qqp/predict.sh 5
bash example-scripts/qqp/data.sh 5 report-qqp-3-5-8.csv
bash example-scripts/qqp/memory.sh 5 8
cp -r ./containers /data/sisa/qqp/3-containers-5-8

cd ./datasets/qqp/
python preprocess_data.py 60000
python prepare_data.py
python update_dataset_file.py 48000 12000
cd ..
cd ..
rm -rf ./containers/
bash example-scripts/qqp/init.sh 5
bash example-scripts/qqp/train.sh 5 16
bash example-scripts/qqp/predict.sh 5
bash example-scripts/qqp/data.sh 5 report-qqp-1-5-16.csv
bash example-scripts/qqp/memory.sh 5 16
cp -r ./containers /data/sisa/qqp/1-containers-5-16

cd ./datasets/qqp/
python preprocess_data.py 60000
python prepare_data.py
python update_dataset_file.py 48000 12000
cd ..
cd ..
rm -rf ./containers/
bash example-scripts/qqp/init.sh 5
bash example-scripts/qqp/train.sh 5 16
bash example-scripts/qqp/predict.sh 5
bash example-scripts/qqp/data.sh 5 report-qqp-2-5-16.csv
bash example-scripts/qqp/memory.sh 5 16
cp -r ./containers /data/sisa/qqp/2-containers-5-16

cd ./datasets/qqp/
python preprocess_data.py 60000
python prepare_data.py
python update_dataset_file.py 48000 12000
cd ..
cd ..
rm -rf ./containers/
bash example-scripts/qqp/init.sh 5
bash example-scripts/qqp/train.sh 5 16
bash example-scripts/qqp/predict.sh 5
bash example-scripts/qqp/data.sh 5 report-qqp-3-5-16.csv
bash example-scripts/qqp/memory.sh 5 16
cp -r ./containers /data/sisa/qqp/3-containers-5-16

cd ./datasets/qqp/
python preprocess_data.py 60000
python prepare_data.py
python update_dataset_file.py 48000 12000
cd ..
cd ..
rm -rf ./containers/
bash example-scripts/qqp/init.sh 5
bash example-scripts/qqp/train.sh 5 32
bash example-scripts/qqp/predict.sh 5
bash example-scripts/qqp/data.sh 5 report-qqp-1-5-32.csv
bash example-scripts/qqp/memory.sh 5 32
cp -r ./containers /data/sisa/qqp/1-containers-5-32

cd ./datasets/qqp/
python preprocess_data.py 60000
python prepare_data.py
python update_dataset_file.py 48000 12000
cd ..
cd ..
rm -rf ./containers/
bash example-scripts/qqp/init.sh 5
bash example-scripts/qqp/train.sh 5 32
bash example-scripts/qqp/predict.sh 5
bash example-scripts/qqp/data.sh 5 report-qqp-2-5-32.csv
bash example-scripts/qqp/memory.sh 5 32
cp -r ./containers /data/sisa/qqp/2-containers-5-32

cd ./datasets/qqp/
python preprocess_data.py 60000
python prepare_data.py
python update_dataset_file.py 48000 12000
cd ..
cd ..
rm -rf ./containers/
bash example-scripts/qqp/init.sh 5
bash example-scripts/qqp/train.sh 5 32
bash example-scripts/qqp/predict.sh 5
bash example-scripts/qqp/data.sh 5 report-qqp-3-5-32.csv
bash example-scripts/qqp/memory.sh 5 32
cp -r ./containers /data/sisa/qqp/3-containers-5-32

cp -r ./results /data/sisa1/qqp/results-runs