
import tqdm
import sys
import os
os.chdir("notebooks")
sys.path.insert(0, "..")

from program_synthesis.analysis.load_results import table_of_accuracies
from program_synthesis.analysis.models_to_analyze import model_labels

for x in model_labels:
    table_of_accuracies(x, pbar=tqdm.tqdm)
