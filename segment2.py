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
COLOR_LIST = ['red_barrel']

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

def showImage(img):
	cv2.imshow('Image',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def showMaskedPart(img, mask):
	rows, cols, channels = img.shape
	res = np.zeros(img.shape, dtype=np.uint8)
	for row in xrange(rows):
		for col in xrange(cols):
			if mask.item(row, col) == True:
				res[row, col, 0] = img.item(row, col, 0)
				res[row, col, 1] = img.item(row, col, 1)
				res[row, col, 2] = img.item(row, col, 2)
			else:
				res[row, col, 0] = 0
				res[row, col, 1] = 0
				res[row, col, 2] = 0
	showImage(res)

def getROIPixels(img, mask):
	rows, cols, channels = img.shape
	res = []
	count = 0
	for row in xrange(rows):
		for col in xrange(cols):
			if mask.item(row, col) == True:
				res.append(img.item(row, col, 0))
				res.append(img.item(row, col, 1))
				res.append(img.item(row, col, 2))
				count = count + 1
	res = np.asarray(res).reshape(count, 3)
	return res

def prob_x_cl(x, mean, covariance):
	constant = 1.0 / ((((2 * math.pi) ** 3) * np.linalg.det(covariance)) ** (1/2.0))
	exponent = -0.5 * np.transpose(x - mean) * np.linalg.inv(covariance) * (x - mean)
	return constant * math.exp(exponent)

GaussianMLEParams = namedtuple('GaussianMLEParams', ['color', 'mean', 'cov'])

def gaussianMLE(training):
	roi = [None] * len(COLOR_LIST)
	mean = [None] * len(COLOR_LIST)
	covariance = [None] * len(COLOR_LIST)

	for idx, color in enumerate(COLOR_LIST):
		roiPixels = np.empty((0,3), dtype=np.uint8)
		for file in training:
			img = cv2.imread(os.path.join(DATA_FOLDER, file))
			# img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
			mask = getImageROIMask(file, color)
			roiPixelsInFile = getROIPixels(img, mask)
			roiPixels = np.concatenate([roiPixels, roiPixelsInFile])
		if roiPixels.shape[0] != 0:
			mean[idx] = calMean(roiPixels)
			covariance[idx] = calCovariance(roiPixels.T)

	model = GaussianMLEParams(color=COLOR_LIST, mean=mean, cov=covariance)
	return model

def gaussianPredict(model, test):
	if len(model.color) == 1:
		for file in test:
			img = cv2.imread(os.path.join(DATA_FOLDER, file))
			rows, cols, channels = img.shape
			for row in xrange(rows):
				for col in xrange(cols):
					x = np.asarray([img.item(row, col, 0), img.item(row, col, 1), img.item(row, col, 2)])
					print prob_x_cl(x, model.mean[0], model.cov[0])
			break

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