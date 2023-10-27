import json
import sys

with open("datasetfile", "r") as f:
    obj = json.load(f)

num_train = int(sys.argv[1])
num_test = int(sys.argv[2])

obj["nb_train"] = num_train
obj["nb_test"] = num_test

with open("datasetfile", "w") as f:
    json.dump(obj, f)