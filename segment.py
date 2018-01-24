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
LOOKUP_TABLE_FOLDER = 'lookupTable'

#######################################################################################################################################

def test():
    testFolder = "Test_Set"
    outputFolder = os.path.join(testFolder, 'OutputFolder')
    segmentedImageFolder = os.path.join(outputFolder, 'SegmentedImages')
    boundingBoxFolder = os.path.join(outputFolder, 'BboxImages')
    g = GmmMLE(COLOR_LIST, DATA_FOLDER, numMixtures=2, covMethod = 'FullCov')
    with open(os.path.join(LOOKUP_TABLE_FOLDER, 'GmmTable'), 'rb') as input:
        table = pickle.load(input)

    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    if not os.path.exists(segmentedImageFolder):
        os.makedirs(segmentedImageFolder)

    if not os.path.exists(boundingBoxFolder):
        os.makedirs(boundingBoxFolder)

    for filename in os.listdir(testFolder):
        if os.path.isfile(os.path.join(testFolder,filename)):
            # read one test image
            img = cv2.imread(os.path.join(testFolder,filename))
            # Your computations here!
            maskedImage, bboxImage, barrelDistance, centroidList = doSegmentation(img, table, g.predictWithLookupTable, DATA_FOLDER, COLOR_LIST)
            print 'Distance for ' + filename + ': ' + str(barrelDistance)
            print 'Centroids for ' + filename + ': ' + str(centroidList) + '\n'
            cv2.imwrite(os.path.join(segmentedImageFolder, filename), maskedImage)
            cv2.imwrite(os.path.join(boundingBoxFolder, filename), bboxImage)

def train():
    # g = GaussianMLE(COLOR_LIST, DATA_FOLDER)
    g = GmmMLE(COLOR_LIST, DATA_FOLDER, numMixtures=2, covMethod = 'FullCov')
    saveLookupTable(g.train, g.getLookupTable, 'GmmTable', DATA_FOLDER)

if __name__ == "__main__":
    doSomeTests(DATA_FOLDER, COLOR_LIST)
    # train()
    # test()