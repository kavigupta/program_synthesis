import argparse
import copy
import math
import random
import sys
import os
import json

import torch

# Add current directory to sys PATH.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.'))

import arguments
import datasets
import models
import tools
from tools import timer, saver
from models.karel_agent import main as rl_main

if __name__ == "__main__":
    rl_main()
