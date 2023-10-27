import os
import sys

dataset_name = sys.argv[1]
shards = sys.argv[2]
slices = sys.argv[3]

cache_dir = "./containers/" + str(shards) + "/cache/"
# files = os.listdir(cache_dir)
# files = list(filter(lambda x: "shard" not in x, files))
# total_memory = 0
# for file in files:
#     file_path = os.path.join(cache_dir, file)
#     res = os.stat(file_path).st_size
#     total_memory += res
total_memory = int(os.popen("du ./containers/" + str(shards) + "/cache/").readlines()[-1].split("\t")[0])

with open("./results/memory_" + dataset_name + "_" + str(shards) + "_" + str(slices), "w") as f:
    f.write(str(total_memory))
