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

def getTrainingTestSplit(fileList):
	n = len(fileList)
	numTraining = int(0.8 * n)

	np.random.shuffle(fileList)
	training, test = fileList[:numTraining], fileList[numTraining:]
	return training, test

def getAllFilesInFolder(folder):
	fileList = []
	for imageName in os.listdir(folder):
		if os.path.isfile(os.path.join(folder, imageName)):
			fileList.append(imageName)
	return fileList

def crossValidatedAlgo(algo, predict, DATA_FOLDER):
	fileList = getAllFilesInFolder(DATA_FOLDER)
	training, test = getTrainingTestSplit(fileList)
	model = algo(training)

	for file in test:
		img = cv2.imread(os.path.join(DATA_FOLDER, file))
		testResultMask = predict(model, file)
		showMaskedPart(img, testResultMask, file)
		showBoundingBoxes(img, testResultMask)
		img = cv2.imread(os.path.join(DATA_FOLDER, file))
		showBestBoundingBox(img, testResultMask)

def trainAllTestAll(algo, predict, DATA_FOLDER, OUT_FOLDER):
	fileList = getAllFilesInFolder(DATA_FOLDER)
	model = algo(fileList)

	if not os.path.exists(OUT_FOLDER):
		os.makedirs(OUT_FOLDER)

	for file in fileList:
		img = cv2.imread(os.path.join(DATA_FOLDER, file))
		testResultMask = predict(model, file)
		# showMaskedPart(img, testResultMask, file)
		# showBoundingBoxes(img, testResultMask)
		img = cv2.imread(os.path.join(DATA_FOLDER, file))
		img = getBestBoundingBox(img, testResultMask)
		cv2.imwrite(os.path.join(OUT_FOLDER, os.path.basename(file)), img)

def trainAllTestAllLookup(lookupFileName, predictLookup, DATA_FOLDER, OUT_FOLDER):
	fileList = getAllFilesInFolder(DATA_FOLDER)
	if not os.path.exists(OUT_FOLDER):
		os.makedirs(OUT_FOLDER)

	LOOKUP_FOLDER = 'lookupTable'

	with open(os.path.join(LOOKUP_FOLDER, lookupFileName), 'rb') as input:
		model = pickle.load(input)

	for file in fileList:
		img = cv2.imread(os.path.join(DATA_FOLDER, file))
		testResultMask = predictLookup(model, file)
		# showMaskedPart(img, testResultMask, file)
		# showBoundingBoxes(img, testResultMask)
		img = cv2.imread(os.path.join(DATA_FOLDER, file))
		img = getBestBoundingBox(img, testResultMask)
		cv2.imwrite(os.path.join(OUT_FOLDER, os.path.basename(file)), img)

def saveLookupTable(algo, lookupTableFunc, filename, DATA_FOLDER):
	fileList = getAllFilesInFolder(DATA_FOLDER)
	model = algo(fileList)

	LOOKUP_FOLDER = 'lookupTable'
	if not os.path.exists(LOOKUP_FOLDER):
		os.makedirs(LOOKUP_FOLDER)

	with open(os.path.join(LOOKUP_FOLDER, filename), 'wb') as output:
		pickle.dump(lookupTableFunc(model), output, pickle.HIGHEST_PROTOCOL)

def plotLookupTable(lookupFileName):
	LOOKUP_FOLDER = 'lookupTable'

	with open(os.path.join(LOOKUP_FOLDER, lookupFileName), 'rb') as input:
		model = pickle.load(input)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	pos = np.where(model==1)
	ax.plot_wireframe(pos[0], pos[1], pos[2], color='black')
	# ax.scatter(pos[0], pos[1], pos[2], c='red')
	plt.show()

def myAlgorithm(img):
	cv2.imshow('image',img)
	return 0,0,0

def test():
	folder = "Test_Set"
	for filename in os.listdir(folder):
		# read one test image
		img = cv2.imread(os.path.join(folder,filename))
		# Your computations here!
		x, y, d = myAlgorithm(img)
		# Display results:
		# (1) Segmented image
		# (2) Barrel bounding box
		# (3) Distance of barrel
		# You may also want to plot and display other diagnostic information
		print filename
		cv2.waitKey(0)
		cv2.destroyAllWindows()