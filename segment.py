import cv2, os
import numpy as np
from roipoly import roipoly
import pylab as pl
import matplotlib.pyplot as plt
import pickle
import Tkinter
import tkMessageBox
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

    # g = GaussianMLE(COLOR_LIST, DATA_FOLDER)
    g = GmmMLE(COLOR_LIST, DATA_FOLDER)

    # crossValidatedAlgo(g.train, g.predict, DATA_FOLDER)

    # trainAllTestAll(g.train, g.predict, DATA_FOLDER, 'outBbox_Gmm_1')

    saveLookupTable(g.train, g.getLookupTable, 'GmmMLE', DATA_FOLDER)
    trainAllTestAllLookup('GmmMLE', g.predictWithLookupTable, DATA_FOLDER, 'outBbox_Gmm_1')

    #######################
    #######################
	# crossValidatedAlgo(gaussianMLE, gaussianPredict)
    # trainAllTestAll(gaussianMLE, gaussianPredict)

    # saveLookupTable(gaussianMLE, getGaussianLookupTable, 'GaussianMLE')
    # plotLookupTable('GaussianMLE')
    # trainAllTestAllLookup('GaussianMLE', gaussianPredictLookup)

    #######

    # crossValidatedAlgo(gmmMLE, gaussianPredict)
    # trainAllTestAll(gmmMLE, gmmPredict)
    # saveLookupTable(gmmMLE, getGmmLookupTable, 'GmmMLE')
    # trainAllTestAllLookup('GmmMLE', gmmPredictLookup)


# Better bounding box statistics - account for the tilt, merging bounding boxes behind objects etc
# Take prior (as opposed to uniform at present) for different colors... 