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

DATA_FOLDER = '2018Proj1_train'
ROI_FOLDER = os.path.join(DATA_FOLDER, 'roi_data')
COLOR_LIST = ['red_barrel', 'white_shine', 'red_nonbarrel']

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

def showMaskedPart(img, mask, imageName='Image'):
	res = np.zeros(img.shape, dtype=np.uint8)
	firstChannel = img[:, :, 0]
	secondChannel = img[:, :, 1]
	thirdChannel = img[:, :, 2]

	firstChannel[~mask] = 0 
	secondChannel[~mask] = 0
	thirdChannel[~mask] = 0

	res[:, :, 0] = firstChannel
	res[:, :, 1] = secondChannel
	res[:, :, 2] = thirdChannel

	showImage(res, imageName)

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

def gaussianPredict(model, test):
	if len(model.color) == 1:
		# threshold = 1e-07 #for RGB
		threshold = 1e-05 #for Y_CR_CB
		for file in test:
			img = cv2.imread(os.path.join(DATA_FOLDER, file))
			img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
			rows, cols, channels = img.shape

			imgSize = np.shape(img)
			res = np.zeros(imgSize[0:2], dtype=bool)

			for row in xrange(rows):
				for col in xrange(cols):
					x = np.asarray([img.item(row, col, 0), img.item(row, col, 1), img.item(row, col, 2)])
					red_barrel_probability = prob_x_cl(x, model.mean[0], model.cov[0], model.covInverse[0])
					# print red_barrel_probability

					if red_barrel_probability > threshold:
						res.itemset((row, col), True)
			
			showMaskedPart(img, res, file)
			# break
	else:
		# threshold = 1e-07 #for RGB
		threshold = 1e-06 #for Y_CR_CB
		for file in test:
			img = cv2.imread(os.path.join(DATA_FOLDER, file))
			img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
			rows, cols, channels = img.shape

			imgSize = np.shape(img)
			res = np.zeros(imgSize[0:2], dtype=bool)

			for row in xrange(rows):
				for col in xrange(cols):
					x = np.asarray([img.item(row, col, 0), img.item(row, col, 1), img.item(row, col, 2)])
					red_barrel_probability = prob_x_cl(x, model.mean[0], model.cov[0], model.covInverse[0])

					max_other_probability = 0
					for c in range(1, len(model.color)):
						this_color_probability = prob_x_cl(x, model.mean[c], model.cov[c], model.covInverse[c])
						if this_color_probability > max_other_probability:
							max_other_probability = this_color_probability

					if red_barrel_probability > max_other_probability and red_barrel_probability > threshold:
						res.itemset((row, col), True)
			
			showMaskedPart(img, res, file)


#######################################################################################################################################

def getTrainingTestSplit(fileList):
	n = len(fileList)
	numTraining = int(0.8 * n)

	np.random.shuffle(fileList)
	training, test = fileList[:numTraining], fileList[numTraining:]
	return training, test

def crossValidatedAlgo(algo, predict):
	folder = DATA_FOLDER
	fileList = []
	for imageName in os.listdir(folder):
		if os.path.isfile(os.path.join(folder, imageName)):
			fileList.append(imageName)

	training, test = getTrainingTestSplit(fileList)
	model = algo(training)
	testResults = predict(model, test)

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
    crossValidatedAlgo(gaussianMLE, gaussianPredict)