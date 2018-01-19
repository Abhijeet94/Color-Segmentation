import cv2, os
import numpy as np
from roipoly import roipoly
import pylab as pl
import pickle
import Tkinter
import tkMessageBox

DATA_FOLDER = '2018Proj1_train'
ROI_FOLDER = os.path.join(DATA_FOLDER, 'roi_data')

def showImage(img):
	cv2.imshow('Image',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def labelledCorrectly():
    result = tkMessageBox.askquestion("Proceed", "Are You Sure?")
    if result == 'yes':
        return True
    else:
        return False

def saveImageROIMask(imgName, color):
	fullImgPath = os.path.join(DATA_FOLDER, imgName)
	roiImgName = imgName + '.pkl'
	fullImgRoiPath = os.path.join(ROI_FOLDER, os.path.join(color, roiImgName))

	if not os.path.exists(ROI_FOLDER):
		os.makedirs(ROI_FOLDER)

	if not os.path.exists(os.path.join(ROI_FOLDER, color)):
		os.makedirs(os.path.join(ROI_FOLDER, color))

	while True:
		img = pl.imread(fullImgPath)

		pl.imshow(img, interpolation='nearest', cmap="Greys")
		pl.colorbar()
		pl.title('Label: ' + str(color))

		# let user draw ROI
		ROI = roipoly(roicolor='g')

		# show the image with the ROI
		pl.imshow(img, interpolation='nearest', cmap="Greys")
		pl.colorbar()
		ROI.displayROI()
		pl.title('The ROI')
		pl.show()

		# show ROI mask
		mask = ROI.getMask(img[:,:,0])
		# pl.imshow(mask, interpolation='nearest', cmap="Greys")
		# pl.title('ROI masks')
		# pl.show()

		if not labelledCorrectly():
			continue

		# print str(np.shape(mask))
		# print str(np.shape(img))

		with open(fullImgRoiPath, 'wb') as output:
			pickle.dump(mask, output, pickle.HIGHEST_PROTOCOL)

		break

def getImageROIMask(imgName, color):
	fullImgPath = os.path.join(DATA_FOLDER, imgName)
	roiImgName = imgName + '.pkl'
	fullImgRoiPath = os.path.join(ROI_FOLDER, os.path.join(color, roiImgName))

	while not os.path.isfile(fullImgRoiPath):
		saveImageROIMask(imgName, color)

	with open(fullImgRoiPath, 'rb') as input:
		mask = pickle.load(input)

	return mask

def getMaskedPixels(img, mask):
	pass

def gaussianMLE(training):
	for file in training:
		img = cv2.imread(os.path.join(DATA_FOLDER, file))
		# img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

		rows, cols, channels = img.shape

		mask = getImageROIMask(file, 'red_barrel')

		# mask = np.array(mask, dtype=np.uint8)
		# res = cv2.bitwise_and(img,img,mask = mask)
		# showImage(res);

		mask = np.array(mask, dtype=np.uint8)
		res = np.array(img[mask], dtype=np.uint8)
		res = cv2.bitwise_and(img,img,mask = mask)
		showImage(res);
		break

def predict(model, test):
	pass

def getTrainingTestSplit(fileList):
	n = len(fileList)
	numTraining = int(0.8 * n)

	np.random.shuffle(fileList)
	training, test = fileList[:numTraining], fileList[numTraining:]
	return training, test

def crossValidatedAlgo(algo):
	folder = DATA_FOLDER
	fileList = []
	for imageName in os.listdir(folder):
		fileList.append(imageName)

	training, test = getTrainingTestSplit(fileList)
	model = algo(training)
	# testResults = predict(model, test)

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

if __name__ == "__main__":
    crossValidatedAlgo(gaussianMLE)