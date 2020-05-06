
import os

import argparse
import json

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input-file", required=True)
parser.add_argument("--output-file", required=True)
parser.add_argument("--start", type=float, required=True)
parser.add_argument("--end", type=float, required=True)

args = parser.parse_args()

assert not os.path.exists(args.output_file)

with open(args.input_file) as f:
    data = json.load(f)

print("Num data samples", len(data))

length = len(data)
indices = list(range(length))
np.random.RandomState(0).shuffle(indices)
valid_indices = set(indices[int(length * args.start) : int(length * args.end)])

invalid_indices = set(range(length)) - valid_indices
for i in invalid_indices:
    data[i] = None

with open(args.output_file, "w") as f:
    json.dump(data, f)