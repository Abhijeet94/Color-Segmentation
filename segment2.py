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

DATA_FOLDER = '2018Proj1_train'
ROI_FOLDER = os.path.join(DATA_FOLDER, 'roi_data')
COLOR_LIST = ['red_barrel', 'white_shine', 'red_nonbarrel', 'black_dark', 'green', 'bluish']

#######################################################################################################################################

retry = True
rois = []

def on_keypress(event, datafilename, img):
	global retry
	global rois
	if event.key == 'n':
		# save 
		imgSize = np.shape(img)
		mask = np.zeros(imgSize[0:2], dtype=bool)
		for roi in rois:
			mask = np.logical_or(mask, roi.getMask2(imgSize))
		np.save(datafilename, mask)
		print("Saving " + datafilename)
		plt.close()
	elif event.key == 'q':
		print("Quitting")
		exit()
	elif event.key == 'r':
		# retry
		print("Retry annotation")
		rois = []
		retry = True
		plt.close()
	elif event.key == 'a':
		# add
		print("Add another annotation")
		retry = True
		plt.close()

def saveImageROIMask(imgName, color):
	global retry
	global rois
	basename, extension = os.path.splitext(imgName)
	fullImgPath = os.path.join(DATA_FOLDER, imgName)
	fullImgRoiPath = os.path.join(ROI_FOLDER, os.path.join(color, basename))

	if not os.path.exists(ROI_FOLDER):
		os.makedirs(ROI_FOLDER)

	if not os.path.exists(os.path.join(ROI_FOLDER, color)):
		os.makedirs(os.path.join(ROI_FOLDER, color))

	img = cv2.imread(fullImgPath)
	print fullImgPath
	rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	rois = []
	retry = True
	while (retry):
		retry = False
		plt.cla()

		# draw region of interest
		plt.imshow(rgbImg, interpolation='none')
		for roi in rois:
			roi.displayROI()
		plt.title('Label color: ' + str(color))
		rois.append(roipoly(roicolor='r')) #let user draw ROI

		fig = plt.gcf()
		fig.canvas.mpl_connect('key_press_event', \
			lambda event: on_keypress(event, fullImgRoiPath, rgbImg))

		plt.cla()
		plt.imshow(rgbImg, interpolation='none')
		for roi in rois:
			roi.displayROI()
		plt.title("press \'n\' to save and go to next picture, \'r\' to retry \n \'q\' to quit, \'a\' to add another region")
		plt.show()

def getImageROIMask(imgName, color):
	basename, extension = os.path.splitext(imgName)
	fullImgPath = os.path.join(DATA_FOLDER, imgName)
	fullImgRoiPath = os.path.join(ROI_FOLDER, os.path.join(color, basename))
	fullImgRoiPathWithExtension = fullImgRoiPath + '.npy'

	while not os.path.isfile(fullImgRoiPathWithExtension):
		saveImageROIMask(imgName, color)

	mask = np.load(fullImgRoiPathWithExtension)

	return mask

#######################################################################################################################################

def calMean(m):
	return m.mean(0)

def calCovariance(m):
	return np.cov(m)

def showImage(img, imageName='Image'):
	cv2.imshow(imageName,img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def getMaskedpart(img, mask):
	image = np.copy(img)
	res = np.zeros(image.shape, dtype=np.uint8)
	firstChannel = image[:, :, 0]
	secondChannel = image[:, :, 1]
	thirdChannel = image[:, :, 2]

	firstChannel[~mask] = 0 
	secondChannel[~mask] = 0
	thirdChannel[~mask] = 0

	res[:, :, 0] = firstChannel
	res[:, :, 1] = secondChannel
	res[:, :, 2] = thirdChannel

	return res

def showMaskedPart(img, mask, imageName='Image'):
	showImage(getMaskedpart(img, mask), imageName)

def getROIPixels(img, mask):
	res = np.zeros((np.sum(mask), 3), dtype=np.uint8)
	firstChannel = img[:, :, 0]
	secondChannel = img[:, :, 1]
	thirdChannel = img[:, :, 2]

	res[:, 0] = firstChannel[mask]
	res[:, 1] = secondChannel[mask]
	res[:, 2] = thirdChannel[mask]
	return res

def getBoundingBoxes(img, mask):
	label_img = label(mask, connectivity=mask.ndim)
	props = regionprops(label_img)

	for prop in props:
		x1, y1, x2, y2 = prop.bbox
		cv2.rectangle(img, (y1, x1), (y2, x2), (255,0,0), 2)

	return img

def showBoundingBoxes(img, mask):
	showImage(getBoundingBoxes(img, mask))

def getBestBoundingBox(img, mask):
	label_img = label(mask, connectivity=mask.ndim)
	props = regionprops(label_img)

	analysis = [(prop.filled_area, prop.extent, i) for i, prop in enumerate(props)]

	totalSumOfArea = sum([a[0] for a in analysis])
	mostOfData = 0.90 * totalSumOfArea
	sortedByArea = sorted(analysis, key=lambda x: x[0], reverse=True)

	indexTillMostData = 0
	sumTillNow = 0
	for i in xrange(len(sortedByArea)):
		sumTillNow = sumTillNow + sortedByArea[i][0]
		if sumTillNow >= mostOfData:
			indexTillMostData = i
			break

	goodBboxData = sortedByArea[0:indexTillMostData + 1]
	sortedByExtent = sorted(goodBboxData, key=lambda x: x[1], reverse=True)

	bestBboxIndex = sortedByExtent[0][2]

	print props[bestBboxIndex].centroid

	x1, y1, x2, y2 = props[bestBboxIndex].bbox
	cv2.rectangle(img, (y1, x1), (y2, x2), (255,0,0), 2)

	return img

def showBestBoundingBox(img, mask):
	showImage(getBestBoundingBox(img, mask))

def getMaskMatchScore(groundtruth, predicted):
	rows, cols = groundtruth.shape
	if rows != predicted.shape[0] or cols != predicted.shape[1]:
		return None

	tp = 0; fp = 0; tn = 0; fn = 0;
	for row in rows:
		for col in cols:
			if groundtruth.item(row, col) == True and predicted.item(row, col) == True:
				tp = tp + 1
			if groundtruth.item(row, col) == False and predicted.item(row, col) == True:
				fp = fp + 1
			if groundtruth.item(row, col) == False and predicted.item(row, col) == False:
				tn = tn + 1
			if groundtruth.item(row, col) == True and predicted.item(row, col) == False:
				fn = fn + 1
	precision = tp / (tp + fp + 0.0)
	recall = tp / (tp + fn + 0.0)
	f_measure = 2 * precision * recall / (precision + recall)
	return f_measure

#######################################################################################################################################

def prob_x_cl_gaussian(x, mean, covariance, covarianceInverse):
	constant = (np.linalg.det(covarianceInverse) ** (1/2.0)) / ((((2 * math.pi) ** 3)) ** (1/2.0))

	exp1 = np.matmul(np.transpose(x - mean), covarianceInverse)
	exp2 = np.matmul(exp1, (x - mean))
	exponent = -0.5 * exp2
	result = constant * math.exp(exponent)
	return result

GaussianMLEParams = namedtuple('GaussianMLEParams', ['color', 'mean', 'cov', 'covInverse'])

def gaussianMLE(training):
	mean = [None] * len(COLOR_LIST)
	covariance = [None] * len(COLOR_LIST)
	covarianceInverse = [None] * len(COLOR_LIST)

	for idx, color in enumerate(COLOR_LIST):
		roiPixels = np.empty((0,3), dtype=np.uint8)
		for file in training:
			img = cv2.imread(os.path.join(DATA_FOLDER, file))
			img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
			mask = getImageROIMask(file, color)
			roiPixelsInFile = getROIPixels(img, mask)
			roiPixels = np.concatenate([roiPixels, roiPixelsInFile])
		if roiPixels.shape[0] != 0:
			mean[idx] = calMean(roiPixels)
			covariance[idx] = calCovariance(roiPixels.T)
			covarianceInverse[idx] = np.linalg.inv(covariance[idx])

	model = GaussianMLEParams(color=COLOR_LIST, mean=mean, cov=covariance, covInverse=covarianceInverse)
	return model

def gaussianPredictHelperSingleGaussian(x, model):
	# threshold = 1e-07 #for RGB
	threshold = 1e-05 #for Y_CR_CB
	red_barrel_probability = prob_x_cl_gaussian(x, model.mean[0], model.cov[0], model.covInverse[0])
	if red_barrel_probability > threshold:
		return True
	else:
		return False

def gaussianPredictHelperManyGaussians(x, model):
	# threshold = 1e-07 #for RGB
	threshold = 0#1e-06 #for Y_CR_CB

	red_barrel_probability = prob_x_cl_gaussian(x, model.mean[0], model.cov[0], model.covInverse[0])

	max_other_probability = 0
	for c in range(1, len(model.color)):
		this_color_probability = prob_x_cl_gaussian(x, model.mean[c], model.cov[c], model.covInverse[c])
		if this_color_probability > max_other_probability:
			max_other_probability = this_color_probability

	if red_barrel_probability > max_other_probability and red_barrel_probability > threshold:
		return True
	else:
		return False

def gaussianPredict(model, file):
	if len(model.color) == 1:
		img = cv2.imread(os.path.join(DATA_FOLDER, file))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
		res = np.apply_along_axis(gaussianPredictHelperSingleGaussian, 2, img, model)
	else:
		img = cv2.imread(os.path.join(DATA_FOLDER, file))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
		res = np.apply_along_axis(gaussianPredictHelperManyGaussians, 2, img, model)
	return res

def gaussianPredictLookupHelper(x, model):
	return model.item(x[0], x[1], x[2])

def gaussianPredictLookup(model, file):
	img = cv2.imread(os.path.join(DATA_FOLDER, file))
	img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
	res = np.apply_along_axis(gaussianPredictLookupHelper, 2, img, model)
	return res

def getGaussianLookupTable(model):
	res = np.zeros((256, 256, 256), dtype=bool)
	x, y, z = res.shape

	for i in xrange(x):
		print i
		for j in xrange(y):
			for k in xrange(z):
				if len(model.color) == 1:
					res.itemset((i, j, k), gaussianPredictHelperSingleGaussian(np.asarray([i, j, k]), model))
				else:
					res.itemset((i, j, k), gaussianPredictHelperManyGaussians(np.asarray([i, j, k]), model))

	return res

#######################################################################################################################################


GmmMLEParams = namedtuple('GmmMLEParams', ['color', 'mean', 'cov', 'covInverse', 'mixtureProbabilities'])

def prob_x_cl_gmm(x, mean, covariance, covarianceInverse, mixtureProbabilities):
	result = 0
	for k in range(len(mean)):
		result = result + mixtureProbabilities[k] * prob_x_cl_gaussian(x, np.transpose(mean[k]), covariance[k], covarianceInverse[k])
	return result

def prob_x_cl_gmm_gaussian(x, mean, covariance, covarianceInverse):
	# constant = (np.linalg.det(covarianceInverse) ** (1/2.0)) / ((((2 * math.pi) ** 3)) ** (1/2.0))
	constant1 = (-3.0/2) * math.log(2 * math.pi)

	detSigmaInv = np.linalg.det(covarianceInverse)
	if detSigmaInv > 0:
		constant2 = (1.0/2) * math.log(detSigmaInv)
	else:
		constant2 = 0

	exp1 = np.matmul(np.transpose(x - mean), covarianceInverse)
	exp2 = np.matmul(exp1, (x - mean))
	exponent = -0.5 * exp2

	result = (exponent + constant2 + constant1)
	return result # returns log of N

def logSumExp(N, pi):
	summedArray = [(N[g] + math.log(pi[g])) for g in range(len(N))]
	maxElement = max(summedArray)

	result = maxElement + math.log(sum([math.exp(summedArray[g] - maxElement) for g in range(len(N))]))
	return result

def gmmPredictHelperSingleGaussian(x, model):
	pass

def gmmPredictHelperManyGaussians(x, model):
	# threshold = 1e-07 #for RGB
	threshold = 0#1e-06 #for Y_CR_CB

	red_barrel_probability = prob_x_cl_gmm(x, model.mean[0], model.cov[0], model.covInverse[0], model.mixtureProbabilities[0])

	max_other_probability = 0
	for c in range(1, len(model.color)):
		this_color_probability = prob_x_cl_gmm(x, model.mean[c], model.cov[c], model.covInverse[c], model.mixtureProbabilities[c])
		if this_color_probability > max_other_probability:
			max_other_probability = this_color_probability

	if red_barrel_probability > max_other_probability and red_barrel_probability > threshold:
		return True
	else:
		return False

def gmmPredict(model, file):
	if len(model.color) == 1:
		img = cv2.imread(os.path.join(DATA_FOLDER, file))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
		res = np.apply_along_axis(gmmPredictHelperSingleGaussian, 2, img, model)
	else:
		img = cv2.imread(os.path.join(DATA_FOLDER, file))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
		res = np.apply_along_axis(gmmPredictHelperManyGaussians, 2, img, model)
	return res

def gmmPredictLookupHelper(x, model):
	return model.item(x[0], x[1], x[2])

def gmmPredictLookup(model, file):
	img = cv2.imread(os.path.join(DATA_FOLDER, file))
	img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
	res = np.apply_along_axis(gmmPredictLookupHelper, 2, img, model)
	return res

def getGmmLookupTable(model):
	res = np.zeros((256, 256, 256), dtype=bool)
	x, y, z = res.shape

	for i in xrange(x):
		print i
		for j in xrange(y):
			for k in xrange(z):
				if len(model.color) == 1:
					res.itemset((i, j, k), gmmPredictHelperSingleGaussian(np.asarray([i, j, k]), model))
				else:
					res.itemset((i, j, k), gmmPredictHelperManyGaussians(np.asarray([i, j, k]), model))

	return res

def initializeEMparameters(k):
	mu =  [255 * np.random.random_sample((1, 3)) for _ in range(k)]
	sigma = [70 * np.identity(3)] * k
	mixProb = [(1.0/k)] * k
	return mu, sigma, mixProb

def EM(X):
	numTry = 2
	k = 3 # Number of mixtures
	n = X.shape[0]
	logLikelihood = 0; maxLikelihood = float("-inf")
	best_mu, best_sigma, best_mixProb = initializeEMparameters(k)

	for _ in xrange(numTry):
		mu, sigma, mixProb = initializeEMparameters(k)
		sigmaInverse = [np.linalg.pinv(s) for s in sigma]
		membership = np.zeros((n, k))
		previousLL = 0
		numSaturation = 0

		while True:

			# Pg 438-439 PRML, Bishop
			# E-step = evaluate membership probabilities using current parameter values
			for i in xrange(membership.shape[0]):
				N_log = [prob_x_cl_gmm_gaussian((X[i, :]).reshape(3, 1), np.transpose(m), s, si) for m, s, si in zip(mu, sigma, sigmaInverse)]
				denominatorLog = logSumExp(N_log, mixProb) # sum(mixProb[g] * N[g] for g in range(len(N)))



				# membership[i, :] = (1.0/denominator) * np.multiply(np.asarray(mixProb), np.exp(np.asarray(N)))
				for j in range(k):
					membership[i, j] = math.exp(math.log(mixProb[j]) + N_log[j] - denominatorLog)
				# print N
				# print membership
			# E-step done
			print 'E-step done'
			#################################################################################

			# M-step = Re-estimate the parameters using current membership probabilities
			Nk = np.sum(membership, axis=0)
			mixProb = (1.0/n) * Nk
			print mixProb

			for j in xrange(k):
				# mu[j] = (1.0/Nk[j]) * np.average(X, axis=0, weights=membership[:, j])
				cumSum = np.zeros((1, 3))
				for i in range(n):
					cumSum = cumSum + membership[i, j] * X[i, :]
				mu[j] = (1.0/Nk[j]) * cumSum
			print mu
			print 'mu calculated'

			for j in xrange(k):
				shiftedX = np.subtract(X, mu[j])
				# XX = np.matmul(np.transpose(shiftedX), shiftedX)
				# sigma[j] = (1.0/Nk[j]) * np.average(XX, axis=0, weights=membership[:, j])
				cumSum = np.zeros((3, 3))
				for i in range(n):
					prod = np.matmul(np.transpose(shiftedX[i, :]), shiftedX[i, :])
					cumSum = cumSum + membership[i, j] * prod
				sigma[j] = (1.0/Nk[j]) * cumSum
				# sigma[j] = (1.0/Nk[j]) * sum([membership[i, j] * np.matmul(np.transpose(shiftedX[i, :]), shiftedX[i, :]) for i in range(n)])
			print sigma
			sigmaInverse = [np.linalg.pinv(s) for s in sigma]
			# M-step done
			#################################################################################

			# Evaluate the log-likelihood
			logLikelihood = 0
			for i in xrange(n):
				N = [prob_x_cl_gmm_gaussian((X[i, :]).reshape(3, 1), np.transpose(m), s, si) for m, s, si in zip(mu, sigma, sigmaInverse)]
				probSum = logSumExp(N, mixProb) #sum(mixProb[g] * N[g] for g in range(len(N)))
				# print probSum
				logLikelihood = logLikelihood + (probSum)
			# Evaluate done
			#################################################################################

			print 'Log-likelihood: ',
			print logLikelihood

			# Check for convergence; Break if converged
			if(abs(previousLL - logLikelihood) < 1 and numSaturation > 1):
				break
			elif(abs(previousLL - logLikelihood) < 1 and numSaturation <= 1):
				numSaturation = numSaturation + 1
			previousLL = logLikelihood
			# Check done

		if(maxLikelihood < logLikelihood):
			maxLikelihood = logLikelihood
			best_mu = mu
			best_sigma = sigma
			best_mixProb = mixProb

	return best_mu, best_sigma, best_mixProb, [np.linalg.pinv(s) for s in best_sigma]

def gmmMLE(training):
	mean = [None] * len(COLOR_LIST)
	covariance = [None] * len(COLOR_LIST)
	mixingProbabilites = [None] * len(COLOR_LIST)
	covarianceInverse = [None] * len(COLOR_LIST)

	for idx, color in enumerate(COLOR_LIST):
		roiPixels = np.empty((0,3), dtype=np.uint8)
		for file in training:
			img = cv2.imread(os.path.join(DATA_FOLDER, file))
			img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
			mask = getImageROIMask(file, color)
			roiPixelsInFile = getROIPixels(img, mask)
			roiPixels = np.concatenate([roiPixels, roiPixelsInFile])
		if roiPixels.shape[0] != 0:
			mu, sigma, mixProb, sigmaInverse = EM(roiPixels)
			mean[idx] = mu
			covariance[idx] = sigma
			mixingProbabilites[idx] = mixProb
			covarianceInverse[idx] = sigmaInverse

	model = GmmMLEParams(color=COLOR_LIST, mean=mean, cov=covariance, covInverse=covarianceInverse, mixtureProbabilities=mixingProbabilites)
	return model

#######################################################################################################################################
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

def crossValidatedAlgo(algo, predict):
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

def trainAllTestAll(algo, predict):
	fileList = getAllFilesInFolder(DATA_FOLDER)
	model = algo(fileList)

	OUT_FOLDER = 'outputBbox_experiment_with_EM'
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

def trainAllTestAllLookup(lookupFileName, predictLookup):
	fileList = getAllFilesInFolder(DATA_FOLDER)
	OUT_FOLDER = 'outputBboxLookup_experiment_with_EM'
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

def saveLookupTable(algo, lookupTableFunc, filename):
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

#######################################################################################################################################

if __name__ == "__main__":
	# crossValidatedAlgo(gaussianMLE, gaussianPredict)
    # trainAllTestAll(gaussianMLE, gaussianPredict)

    # saveLookupTable(gaussianMLE, getGaussianLookupTable, 'GaussianMLE')
    # plotLookupTable('GaussianMLE')
    # trainAllTestAllLookup('GaussianMLE', gaussianPredictLookup)

    #######

    # crossValidatedAlgo(gmmMLE, gaussianPredict)
    # trainAllTestAll(gmmMLE, gmmPredict)
    saveLookupTable(gmmMLE, getGmmLookupTable, 'GmmMLE')
    trainAllTestAllLookup('GmmMLE', gmmPredictLookup)
# Better bounding box statistics - account for the tilt, merging bounding boxes behind objects etc
# Take prior (as opposed to uniform at present) for different colors... 