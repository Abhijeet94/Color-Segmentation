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


GaussianMLEParams = namedtuple('GaussianMLEParams', ['color', 'mean', 'cov', 'covInverse'])
GmmMLEParams = namedtuple('GmmMLEParams', ['color', 'mean', 'cov', 'covInverse', 'mixtureProbabilities'])

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

def saveImageROIMask(imgName, color, DATA_FOLDER):
	global retry
	global rois
	ROI_FOLDER = os.path.join(DATA_FOLDER, 'roi_data')
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

def getImageROIMask(imgName, color, DATA_FOLDER):
	ROI_FOLDER = os.path.join(DATA_FOLDER, 'roi_data')
	basename, extension = os.path.splitext(imgName)
	fullImgPath = os.path.join(DATA_FOLDER, imgName)
	fullImgRoiPath = os.path.join(ROI_FOLDER, os.path.join(color, basename))
	fullImgRoiPathWithExtension = fullImgRoiPath + '.npy'

	while not os.path.isfile(fullImgRoiPathWithExtension):
		saveImageROIMask(imgName, color, DATA_FOLDER)

	mask = np.load(fullImgRoiPathWithExtension)

	return mask

#######################################################################################################################################

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
	centroidList = []
	dimensionList = []
	areaList = []

	label_img = label(mask, connectivity=mask.ndim)
	props = regionprops(label_img)

	analysis = [(prop.filled_area, prop.extent, i) for i, prop in enumerate(props)]

	if len(analysis) > 0:
		totalSumOfArea = sum([a[0] for a in analysis])
		mostOfData = 0.70 * totalSumOfArea
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
		bestBboxArea = props[bestBboxIndex].filled_area
		bestBboxExtent = props[bestBboxIndex].extent

		if bestBboxExtent > 0.45 and bestBboxArea > 1200:

			x1, y1, x2, y2 = props[bestBboxIndex].bbox
			centroidCoord = ((y1+(y2-y1)/2.0), (x1+(x2-x1)/2.0))
			if bestBboxExtent < 0.9 and bestBboxExtent > 0.8:
				centroidCoord = (centroidCoord[0], centroidCoord[1] + 0.03 * (abs(x2-x1)))

			centroidList.append(centroidCoord)
			dimensionList.append((x1, y1, x2, y2))
			areaList.append(props[bestBboxIndex].filled_area)

			# print bestBboxArea
			# print ((x2-x1) * 1.0)/(y2-y1)
			# print bestBboxExtent
			# print ''

			for ii in range(1, len(sortedByExtent)):
				if sortedByExtent[ii][1] > 0.85 * bestBboxExtent and sortedByExtent[ii][0] > 0.4 * bestBboxArea:
					x1, y1, x2, y2 = props[sortedByExtent[ii][2]].bbox
					dimensionList.append((x1, y1, x2, y2))
					areaList.append(props[sortedByExtent[ii][2]].filled_area)

					centroidCoord = ((y1+(y2-y1)/2.0), (x1+(x2-x1)/2.0))
					if sortedByExtent[ii][1] < 0.9 and sortedByExtent[ii][1] > 0.8:
						centroidCoord = (centroidCoord[0], centroidCoord[1] + 0.03 * (abs(x2-x1)))
					centroidList.append(centroidCoord)

		# Merge bounding boxes belonging to the same barrel but separated due to objects in front of the barrel
		# With the logic that any mergings will be performed with similar extent bounding boxes
		numDeletions = 0
		for i in range(len(centroidList)):
			i = i - numDeletions
			if i+1 < len(centroidList):
				x11, y11, x12, y12 = dimensionList[i]
				extentOf1 = (areaList[i]*1.0)/ (abs(y12 - y11) * abs(x12 - x11))

				x21, y21, x22, y22 = dimensionList[i+1]
				extentOf2 = (areaList[i+1]*1.0)/ (abs(y22 - y21) * abs(x22 - x21))

				combinedExtent = (areaList[i]*1.0 + areaList[i+1])/(abs(max(y22, y12) - min(y21, y11)) * abs(max(x22, x12) - min(x21, x11)))

				if abs(combinedExtent-extentOf1) < 0.2 and abs(combinedExtent-extentOf2) < 0.2 and abs(extentOf1 - extentOf2) < 0.1:
					areaList[i] = areaList[i]*1.0 + areaList[i+1]
					dimensionList[i] = (min(x21, x11), min(y21, y11), max(x22, x12), max(y22, y12))
					centroidList[i] = ((dimensionList[i][1]+(dimensionList[i][3]-dimensionList[i][1])/2.0), (dimensionList[i][0]+(dimensionList[i][2]-dimensionList[i][0])/2.0))
					centroidCoordToShow = (int(round(centroidList[i][0])), int(round(centroidList[i][1])))
					cv2.circle(img, centroidCoordToShow, 3, (255,0,0), -1)
					cv2.rectangle(img, (dimensionList[i][1], dimensionList[i][0]), (dimensionList[i][3], dimensionList[i][2]), (255,0,0), 2)

					del areaList[i+1]
					del dimensionList[i+1]
					del centroidList[i+1]

					numDeletions = numDeletions + 1

				else:
					x1, y1, x2, y2 = dimensionList[i]
					centroidCoord = centroidList[i]
					centroidCoordToShow = (int(round(centroidCoord[0])), int(round(centroidCoord[1])))
					cv2.circle(img, centroidCoordToShow, 3, (255,0,0), -1)
					cv2.rectangle(img, (y1, x1), (y2, x2), (255,0,0), 2)
			else:
				x1, y1, x2, y2 = dimensionList[i]
				centroidCoord = centroidList[i]
				centroidCoordToShow = (int(round(centroidCoord[0])), int(round(centroidCoord[1])))
				cv2.circle(img, centroidCoordToShow, 3, (255,0,0), -1)
				cv2.rectangle(img, (y1, x1), (y2, x2), (255,0,0), 2)


	return img, dimensionList, centroidList, areaList

def showBestBoundingBox(img, mask):
	showImage(getBestBoundingBox(img, mask)[0])

def getMaskMatchScore(groundtruth, predicted):
	rows, cols = groundtruth.shape
	if rows != predicted.shape[0] or cols != predicted.shape[1]:
		return None

	tp = 0; fp = 0; tn = 0; fn = 0;
	for row in range(rows):
		for col in range(cols):
			if groundtruth.item(row, col) == True and predicted.item(row, col) == True:
				tp = tp + 1
			if groundtruth.item(row, col) == False and predicted.item(row, col) == True:
				fp = fp + 1
			if groundtruth.item(row, col) == False and predicted.item(row, col) == False:
				tn = tn + 1
			if groundtruth.item(row, col) == True and predicted.item(row, col) == False:
				fn = fn + 1
	if tp + fp > 0:
		precision = tp / (tp + fp + 0.0)
	else:
		precision = 0

	if tp + fn > 0:
		recall = tp / (tp + fn + 0.0)
	else:
		recall = 0

	if precision + recall > 0:
		f_measure = 2 * precision * recall / (precision + recall)
	else:
		f_measure = 0
	return f_measure, recall