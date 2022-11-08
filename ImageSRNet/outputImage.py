import math
import os

import torch
from matplotlib import pyplot as plt
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import json
import _pickle as pickle

from minis_sr_utils import DataSetManager
from cnn_interpretation import Minist_SR,Args

if __name__ == "__main__":
    fn = f'CGPIndiva/1_0_indiv/bestIndiv.pkl'
    # fn = 'CGPIndiva/0_0bestIndiv.pkl'
    with open(fn, 'rb') as f:
        indiv = pickle.load(f)

    result = DataSetManager(indiv)
    result.getResultImageComp(81,1)
    result.getResultJson()
