import cv2, os
import numpy as np
from roipoly import roipoly
import pylab as pl
import matplotlib.pyplot as plt
import pickle
import math
from collections import namedtuple
from skimage import data, util
from skimage.measure import label, regionprops
from mpl_toolkits.mplot3d import Axes3D

from utils import *
from runAlgoUtils import *
from GaussianMLE import GaussianMLE
from GmmMLE import GmmMLE

#######################################################################################################################################

DATA_FOLDER = '2018Proj1_train'
COLOR_LIST = ['red_barrel', 'white_shine', 'red_nonbarrel', 'black_dark', 'green', 'bluish']

#######################################################################################################################################

if __name__ == "__main__":

    g = GaussianMLE(COLOR_LIST, DATA_FOLDER)
    # g = GmmMLE(COLOR_LIST, DATA_FOLDER)

    crossValidatedAlgo(g.train, g.predict, DATA_FOLDER, 'tempModelGaussian11.pkl')

    # trainAllTestAll(g.train, g.predict, DATA_FOLDER, 'outBbox_Gmm_1')

    # saveLookupTable(g.train, g.getLookupTable, 'Gaussian_test_temp1', DATA_FOLDER)
    # trainAllTestAllLookup('Gaussian_test_temp1', g.predictWithLookupTable, DATA_FOLDER, 'outBbox_Gausian_test_3')
