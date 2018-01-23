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
from scipy.stats import multivariate_normal
from utils import *

class GaussianMLE:

	def __init__(self, colorList, dataFolder):
		self.COLOR_LIST = colorList
		self.DATA_FOLDER = dataFolder

	def prob_x_cl_gaussian(self, x, mean, covariance, covarianceInverse):
		constant = (np.linalg.det(covarianceInverse) ** (1/2.0)) / ((((2 * math.pi) ** 3)) ** (1/2.0))

		exp1 = np.matmul(np.transpose(x - mean), covarianceInverse)
		exp2 = np.matmul(exp1, (x - mean))
		exponent = -0.5 * exp2
		result = constant * math.exp(exponent)
		return result

	def multivariateNormalPdf(self, X, mean, covariance, covarianceInverse):
		constant1 = (-3.0/2) * math.log(2 * math.pi)

		detSigmaInv = np.linalg.det(covarianceInverse)
		if detSigmaInv > 0:
			constant2 = (1.0/2) * math.log(detSigmaInv)
		else:
			constant2 = 0

		X = np.subtract(X, mean)
		exponent = (-0.5) * np.sum(np.multiply(np.matmul(X, covarianceInverse), X), axis = X.ndim - 1)

		result = (exponent + constant2 + constant1)
		return result

	def gaussianPredictHelperSingleGaussian(self, x, model):
		# threshold = 1e-07 #for RGB
		threshold = 1e-05 #for Y_CR_CB
		red_barrel_probability = self.prob_x_cl_gaussian(x, model.mean[0], model.cov[0], model.covInverse[0])
		if red_barrel_probability > threshold:
			return True
		else:
			return False

	def gaussianPredictHelperManyGaussians(self, x, model):
		# threshold = 1e-07 #for RGB
		threshold = 0#1e-06 #for Y_CR_CB

		red_barrel_probability = self.prob_x_cl_gaussian(x, model.mean[0], model.cov[0], model.covInverse[0])

		max_other_probability = 0
		for c in range(1, len(model.color)):
			this_color_probability = self.prob_x_cl_gaussian(x, model.mean[c], model.cov[c], model.covInverse[c])
			if this_color_probability > max_other_probability:
				max_other_probability = this_color_probability

		if red_barrel_probability > max_other_probability and red_barrel_probability > threshold:
			return True
		else:
			return False

	def predict(self, model, file):
		if len(model.color) == 1:
			# threshold = 1e-07 #for RGB
			threshold = 1e-05 #for Y_CR_CB

			img = cv2.imread(os.path.join(self.DATA_FOLDER, file))
			img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
			res = np.apply_along_axis(self.gaussianPredictHelperSingleGaussian, 2, img, model)
		else:
			threshold = -30
			img = cv2.imread(os.path.join(self.DATA_FOLDER, file))
			img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
			# res = np.apply_along_axis(self.gaussianPredictHelperManyGaussians, 2, img, model)

			bigMat = np.zeros((img.shape[0], img.shape[1], len(model.color)))
			for c, color in enumerate(model.color):
				bigMat[:, :, c] = self.multivariateNormalPdf(img, model.mean[c], model.cov[c], model.covInverse[c])
			res = np.argmax(bigMat, axis = 2)
			res = res == 0

			# np.set_printoptions(threshold = np.inf)
			# print bigMat[res][1:500]
			# np.set_printoptions(threshold = 1000)

			resThreshold = np.amax(bigMat, axis = 2) > threshold
			res = np.logical_and(res, resThreshold)

		return res

	def gaussianPredictLookupHelper(self, x, model):
		return model.item(x[0], x[1], x[2])

	def predictWithLookupTable(self, model, file):
		img = cv2.imread(os.path.join(self.DATA_FOLDER, file))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
		res = np.apply_along_axis(self.gaussianPredictLookupHelper, 2, img, model)
		# print 'oioi'
		return res

	def getLookupTable(self, model):
		if len(model.color) == 1:
			res = np.zeros((256, 256, 256), dtype=bool)
			x, y, z = res.shape
			for i in xrange(x):
				print i
				for j in xrange(y):
					for k in xrange(z):
						if len(model.color) == 1:
							res.itemset((i, j, k), self.gaussianPredictHelperSingleGaussian(np.asarray([i, j, k]), model))
						# else:
						# 	res.itemset((i, j, k), self.gaussianPredictHelperManyGaussians(np.asarray([i, j, k]), model))
		else:
			threshold = -30
			res = np.transpose(np.indices((256, 256, 256)), (1, 2, 3, 0))

			bigMat = np.zeros((res.shape[0], res.shape[1], res.shape[2], len(model.color)))
			for c, color in enumerate(model.color):
				bigMat[:, :, :, c] = self.multivariateNormalPdf(res, model.mean[c], model.cov[c], model.covInverse[c])
			res = np.argmax(bigMat, axis = 3)
			res = res == 0

			resThreshold = np.amax(bigMat, axis = 3) > threshold
			res = np.logical_and(res, resThreshold)

		return res

	def train(self, training):
		mean = [None] * len(self.COLOR_LIST)
		covariance = [None] * len(self.COLOR_LIST)
		covarianceInverse = [None] * len(self.COLOR_LIST)

		for idx, color in enumerate(self.COLOR_LIST):
			roiPixels = np.empty((0,3), dtype=np.uint8)
			for file in training:
				img = cv2.imread(os.path.join(self.DATA_FOLDER, file))
				img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
				mask = getImageROIMask(file, color, self.DATA_FOLDER)
				roiPixelsInFile = getROIPixels(img, mask)
				roiPixels = np.concatenate([roiPixels, roiPixelsInFile])
			if roiPixels.shape[0] != 0:
				mean[idx] = calMean(roiPixels)
				covariance[idx] = calCovariance(roiPixels.T)
				covarianceInverse[idx] = np.linalg.inv(covariance[idx])

		model = GaussianMLEParams(color=self.COLOR_LIST, mean=mean, cov=covariance, covInverse=covarianceInverse)
		return model