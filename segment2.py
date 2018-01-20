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

DATA_FOLDER = '2018Proj1_train'
ROI_FOLDER = os.path.join(DATA_FOLDER, 'roi_data')
COLOR_LIST = ['red_barrel', 'white_shine', 'red_nonbarrel', 'black_dark', 'green']

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

def prob_x_cl(x, mean, covariance, covarianceInverse):
	constant = 1.0 / ((((2 * math.pi) ** 3) * np.linalg.det(covariance)) ** (1/2.0))

	exp1 = np.matmul(np.transpose(x - mean), covarianceInverse)
	exp2 = np.matmul(exp1, (x - mean))
	exponent = -0.5 * exp2
	return constant * math.exp(exponent)

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
	red_barrel_probability = prob_x_cl(x, model.mean[0], model.cov[0], model.covInverse[0])
	if red_barrel_probability > threshold:
		return True
	else:
		return False

def gaussianPredictHelperManyGaussians(x, model):
	# threshold = 1e-07 #for RGB
	threshold = 0#1e-06 #for Y_CR_CB

	red_barrel_probability = prob_x_cl(x, model.mean[0], model.cov[0], model.covInverse[0])

	max_other_probability = 0
	for c in range(1, len(model.color)):
		this_color_probability = prob_x_cl(x, model.mean[c], model.cov[c], model.covInverse[c])
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

def showBoundingBoxes(img, mask, suppressShow = False):
	label_img = label(mask, connectivity=mask.ndim)
	props = regionprops(label_img)

	for prop in props:
		x1, y1, x2, y2 = prop.bbox
		cv2.rectangle(img, (y1, x1), (y2, x2), (255,0,0), 2)

	if not suppressShow:
		showImage(img)
	return img

def showBestBoundingBox(img, mask, suppressShow = False):
	label_img = label(mask, connectivity=mask.ndim)
	props = regionprops(label_img)

	analysis = [(prop.filled_area, prop.extent, i) for i, prop in enumerate(props)]

	totalSumOfArea = sum([a[0] for a in analysis])
	mostOfData = 0.95 * totalSumOfArea
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

	if not suppressShow:
		showImage(img)
	return img

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

	OUT_FOLDER = 'outputBbox'
	if not os.path.exists(OUT_FOLDER):
		os.makedirs(OUT_FOLDER)

	for file in fileList:
		img = cv2.imread(os.path.join(DATA_FOLDER, file))
		testResultMask = predict(model, file)
		# showMaskedPart(img, testResultMask, file)
		showBoundingBoxes(img, testResultMask, True)
		img = cv2.imread(os.path.join(DATA_FOLDER, file))
		img = showBestBoundingBox(img, testResultMask, True)
		cv2.imwrite(os.path.join(OUT_FOLDER, os.path.basename(file)), img)

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
    trainAllTestAll(gaussianMLE, gaussianPredict)