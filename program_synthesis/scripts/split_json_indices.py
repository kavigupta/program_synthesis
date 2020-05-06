
import os

import argparse
import json

import numpy as np

from datasets.karel.mutation import KarelOutputRefExampleMutator

parser = argparse.ArgumentParser()
parser.add_argument("--input-file", required=True)
parser.add_argument("--output-file", required=True)
parser.add_argument("--start", type=float, required=True)
parser.add_argument("--end", type=float, required=True)

args = parser.parse_args()

assert not os.path.exists(args.output_file)

with open(args.input_file) as f:
    data = json.load(f)

length = len(data)
print("Num data samples", length)

valid_indices = KarelOutputRefExampleMutator.valid_indices(len(data), args.start, args.end)

invalid_indices = set(range(length)) - valid_indices
for i in invalid_indices:
    data[i] = None

with open(args.output_file, "w") as f:
    json.dump(data, f)