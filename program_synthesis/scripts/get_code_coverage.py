
import sys

import argparse
import tqdm
import json

import multiprocessing

sys.path.insert(0, ".")

from program_synthesis.datasets.dataset import KarelTorchDataset
from program_synthesis.datasets.executor import KarelExecutor

parser = argparse.ArgumentParser()
parser.add_argument('--data')
parser.add_argument('--predictions')
parser.add_argument('--coverage-out')
args = parser.parse_args()

data = KarelTorchDataset(args.data)

with open(args.predictions) as f:
    results = json.load(f)

executor = KarelExecutor()
def func(t):
    return executor.gather_coverage(*t)

with multiprocessing.Pool() as p:
    coverages = list(tqdm.tqdm(p.imap(func, zip(data, results), chunksize=1000), total=len(results)))

with open(args.coverage_out, "w") as f:
    json.dump(coverages, f)
