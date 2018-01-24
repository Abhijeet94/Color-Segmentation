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
from GaussianMLE import GaussianMLE
from GmmMLE import GmmMLE

def getTrainingTestSplit(fileList):
	n = len(fileList)
	numTraining = int(0.9 * n)

	np.random.shuffle(fileList)
	training, test = fileList[:numTraining], fileList[numTraining:]
	return training, test

def getAllFilesInFolder(folder):
	fileList = []
	for imageName in os.listdir(folder):
		if os.path.isfile(os.path.join(folder, imageName)):
			fileList.append(imageName)
	return fileList

def crossValidatedAlgo(algo, predict, DATA_FOLDER, SAVED_MODEL=None):
	fileList = getAllFilesInFolder(DATA_FOLDER)
	training, test = getTrainingTestSplit(fileList)

	if SAVED_MODEL != None and os.path.isfile(SAVED_MODEL):
		with open(SAVED_MODEL, 'rb') as input:
			model = pickle.load(input)
	else:
		model = algo(training)

	if SAVED_MODEL != None:
		with open(SAVED_MODEL, 'wb') as output:
			pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

	finalScore = 0
	for file in test:
		img = cv2.imread(os.path.join(DATA_FOLDER, file))
		testResultMask = predict(model, img)

		groundTruth = getImageROIMask(file, 'red_barrel', DATA_FOLDER)
		# print 'F-measure', 
		fmeasure, recall = getMaskMatchScore(groundTruth, testResultMask)
		# print fmeasure

		finalScore = finalScore + fmeasure

		showMaskedPart(img, testResultMask, file)
		showBoundingBoxes(img, testResultMask)
		img = cv2.imread(os.path.join(DATA_FOLDER, file))
		showBestBoundingBox(img, testResultMask)
	print finalScore/len(test)

def trainAllTestAll(algo, predict, DATA_FOLDER, OUT_FOLDER):
	fileList = getAllFilesInFolder(DATA_FOLDER)
	model = algo(fileList)

	if not os.path.exists(OUT_FOLDER):
		os.makedirs(OUT_FOLDER)

	for file in fileList:
		img = cv2.imread(os.path.join(DATA_FOLDER, file))
		testResultMask = predict(model, img)
		# showMaskedPart(img, testResultMask, file)
		# showBoundingBoxes(img, testResultMask)
		img = cv2.imread(os.path.join(DATA_FOLDER, file))
		img = getBestBoundingBox(img, testResultMask)[0]
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
		testResultMask = predictLookup(model, img)
		# showMaskedPart(img, testResultMask, file)
		# showBoundingBoxes(img, testResultMask)
		img = cv2.imread(os.path.join(DATA_FOLDER, file))
		img = getBestBoundingBox(img, testResultMask)[0]
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

###########################################################################################

def predictWithLookupTable(table, filePath, predictFunction, DATA_FOLDER, groundTruthAvailable=False):
	img = cv2.imread(filePath)
	resultantMask = predictFunction(table, img)

	fmeasure = -1
	recall = -1
	if groundTruthAvailable:
		groundTruth = getImageROIMask(os.path.basename(filePath), 'red_barrel', DATA_FOLDER)
		fmeasure, recall = getMaskMatchScore(groundTruth, resultantMask)

	img, dimensionList, centroidList, areaList = getBestBoundingBox(img, resultantMask)

	return img, dimensionList, centroidList, fmeasure, recall

def predictWithModel(model, filePath, predictFunction, DATA_FOLDER, groundTruthAvailable=False):
	img = cv2.imread(filePath)
	resultantMask = predictFunction(model, img)

	fmeasure = -1
	recall = -1
	if groundTruthAvailable:
		groundTruth = getImageROIMask(os.path.basename(filePath), 'red_barrel', DATA_FOLDER)
		fmeasure, recall = getMaskMatchScore(groundTruth, resultantMask)

	img, dimensionList, centroidList, areaList = getBestBoundingBox(img, resultantMask)

	return img, dimensionList, centroidList, areaList, fmeasure, recall

def crossValidateNumMixtures(DATA_FOLDER, COLOR_LIST):
	fileList = getAllFilesInFolder(DATA_FOLDER)

	k_range = [2, 3, 4, 5, 6]
	fmeasureList = [0] * len(k_range)
	recallList = [0] * len(k_range)
	for ki, k in enumerate(k_range):
		numIterations = 10
		for it in range(numIterations):
			training, test = getTrainingTestSplit(fileList)
			g = GmmMLE(COLOR_LIST, DATA_FOLDER, numMixtures = k)
			model = g.train(training)

			cumulativeFMeasure = 0
			cumulativeRecall = 0
			for file in test:
				img, dimensionList, centroidList, areaList, fmeasure, recall = predictWithModel(model, os.path.join(DATA_FOLDER, file), g.predict, DATA_FOLDER, True)
				cumulativeFMeasure = cumulativeFMeasure + fmeasure
				cumulativeRecall = cumulativeRecall + recall
			fmeasureList[ki] = fmeasureList[ki] + cumulativeFMeasure/len(test)
			recallList[ki] = recallList[ki] + cumulativeRecall/len(test)
		fmeasureList[ki] = fmeasureList[ki]/numIterations
		recallList[ki] = recallList[ki]/numIterations

		print k
		print fmeasureList[ki]
		print recallList[ki]

	print 'Final - '
	print k_range
	print fmeasureList
	print recallList

# 2
# 0.881265483179
# 0.961660081236

# 3
# 0.814542589476
# 0.963764115267

# 4
# 0.737993912978
# 0.962659711823

# 5
# 0.735246303275
# 0.95209516667

def singleGaussianScore(DATA_FOLDER, COLOR_LIST):
	COLOR_LIST = [COLOR_LIST[0]]
	fileList = getAllFilesInFolder(DATA_FOLDER)
	numIterations = 10
	cumulativeFMeasure = 0
	cumulativeRecall = 0
	count = 0
	for it in range(numIterations):
		training, test = getTrainingTestSplit(fileList)
		g = GaussianMLE(COLOR_LIST, DATA_FOLDER)
		model = g.train(training)

		for file in test:
			img, dimensionList, centroidList, areaList, fmeasure, recall = predictWithModel(model, os.path.join(DATA_FOLDER, file), g.predict, DATA_FOLDER, True)
			cumulativeFMeasure = cumulativeFMeasure + fmeasure
			cumulativeRecall = cumulativeRecall + recall
			count = count + 1
	print cumulativeFMeasure/count
	print  cumulativeRecall/count

# 0.702075338927
# 0.65240721917

def multiGaussianScore(DATA_FOLDER, COLOR_LIST):
	fileList = getAllFilesInFolder(DATA_FOLDER)
	numIterations = 10
	cumulativeFMeasure = 0
	cumulativeRecall = 0
	count = 0
	for it in range(numIterations):
		training, test = getTrainingTestSplit(fileList)
		g = GaussianMLE(COLOR_LIST, DATA_FOLDER)
		model = g.train(training)

		for file in test:
			img, dimensionList, centroidList, areaList, fmeasure, recall = predictWithModel(model, os.path.join(DATA_FOLDER, file), g.predict, DATA_FOLDER, True)
			cumulativeFMeasure = cumulativeFMeasure + fmeasure
			cumulativeRecall = cumulativeRecall + recall
			count = count + 1
	print cumulativeFMeasure/count
	print  cumulativeRecall/count

# Gaussians

# Y Cr cb
# 0.88226573548
# 0.959098633618

# RGB
# 0.865992640822
# 0.927456024643

def trainBarrelDistanceModel(DATA_FOLDER):
	fileList = getAllFilesInFolder(DATA_FOLDER)
	X = []
	Y = []
	for file in fileList:
		groundTruthRed = getImageROIMask(file, 'red_barrel', DATA_FOLDER)
		distanceStr = file.split('.')[0]

		if len(distanceStr.split('_')) == 1:
			barrelArea = np.sum(groundTruthRed)
			X.append(1.0 / (barrelArea ** (0.5)))
			Y.append(int(distanceStr))

	nX = np.asarray(X).reshape(len(X), 1)
	nX = np.hstack((nX, np.ones((nX.shape[0], 1))))
	nY = np.asarray(Y)

	# slope = (1.0/np.matmul(nX.T, nX)) * np.matmul(nX.T, nY)
	slope = np.matmul(np.linalg.inv(np.matmul(nX.T, nX)), np.matmul(nX.T, nY))
	print slope

def testBarrelDistanceModel(DATA_FOLDER):
	fileList = getAllFilesInFolder(DATA_FOLDER)
	training, test = getTrainingTestSplit(fileList)
	X = []
	Y = []
	for file in training:
		groundTruthRed = getImageROIMask(file, 'red_barrel', DATA_FOLDER)
		distanceStr = file.split('.')[0]

		if len(distanceStr.split('_')) == 1:
			barrelArea = np.sum(groundTruthRed)
			X.append(1.0 / (barrelArea ** (0.5)))
			Y.append(int(distanceStr))

	nX = np.asarray(X).reshape(len(X), 1)
	nX = np.hstack((nX, np.ones((nX.shape[0], 1))))
	nY = np.asarray(Y)

	# plt.plot(X, Y, 'ro')
	# # plt.axis([0, 6, 0, 20])
	# plt.show()


	slope = np.matmul(np.linalg.inv(np.matmul(nX.T, nX)), np.matmul(nX.T, nY))

	for file in test:
		groundTruthRed = getImageROIMask(file, 'red_barrel', DATA_FOLDER)
		distanceStr = file.split('.')[0]

		if len(distanceStr.split('_')) == 1:
			print 'Actual distance: ' + distanceStr
			barrelArea = np.sum(groundTruthRed)
			x = np.ones((2, 1))
			x[0, 0] = 1.0 / (barrelArea ** (0.5))
			print 'Predicted distance: ' + str(np.dot(slope,x))

def calBarrelDistance(areaList):
	dist = []
	slope = [4.70917397e+02, 2.30127587e-01]
	for area in areaList:
		x = np.ones((2, 1))
		x[0, 0] = 1.0 / (area ** (0.5))
		dist.append(np.dot(slope,x)[0].tolist())
	return dist

###########################################################################################

def doSomeTests(DATA_FOLDER, COLOR_LIST):
	# g = GaussianMLE(COLOR_LIST, DATA_FOLDER)
    # g = GmmMLE(COLOR_LIST, DATA_FOLDER, numMixtures=2, covMethod = 'FullCov')

    # crossValidatedAlgo(g.train, g.predict, DATA_FOLDER, 'tempModelGmm2.pkl')

    # trainAllTestAll(g.train, g.predict, DATA_FOLDER, 'outBbox_Gmm_1')

    # saveLookupTable(g.train, g.getLookupTable, 'GmmTable', DATA_FOLDER)
    # trainAllTestAllLookup('Gaussian_test_temp1', g.predictWithLookupTable, DATA_FOLDER, 'outBbox_Gausian_test_3')
    # crossValidateNumMixtures(DATA_FOLDER, COLOR_LIST)
    # singleGaussianScore(DATA_FOLDER, COLOR_LIST)
    # multiGaussianScore(DATA_FOLDER, COLOR_LIST)
    # testBarrelDistanceModel(DATA_FOLDER)
    # trainBarrelDistanceModel(DATA_FOLDER)
    plotLookupTable('GmmTable')

###########################################################################################

def doSegmentation(img, table, lookupTablePredictFunc, DATA_FOLDER, COLOR_LIST):
	imgCopy = img.copy()
	testResultMask = lookupTablePredictFunc(table, img)
	bboxImage, dimensionList, centroidList, areaList = getBestBoundingBox(img, testResultMask)
	barrelDistance = calBarrelDistance(areaList)
	return getMaskedpart(imgCopy, testResultMask), bboxImage, barrelDistance, centroidList

