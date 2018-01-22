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
			img = cv2.imread(os.path.join(self.DATA_FOLDER, file))
			img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
			res = np.apply_along_axis(self.gaussianPredictHelperSingleGaussian, 2, img, model)
		else:
			img = cv2.imread(os.path.join(self.DATA_FOLDER, file))
			img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
			res = np.apply_along_axis(self.gaussianPredictHelperManyGaussians, 2, img, model)
		return res

	def gaussianPredictLookupHelper(self, x, model):
		return model.item(x[0], x[1], x[2])

	def predictWithLookupTable(self, model, file):
		img = cv2.imread(os.path.join(self.DATA_FOLDER, file))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
		res = np.apply_along_axis(self.gaussianPredictLookupHelper, 2, img, model)
		return res

	def getLookupTable(self, model):
		res = np.zeros((256, 256, 256), dtype=bool)
		x, y, z = res.shape

		for i in xrange(x):
			print i
			for j in xrange(y):
				for k in xrange(z):
					if len(model.color) == 1:
						res.itemset((i, j, k), self.gaussianPredictHelperSingleGaussian(np.asarray([i, j, k]), model))
					else:
						res.itemset((i, j, k), self.gaussianPredictHelperManyGaussians(np.asarray([i, j, k]), model))
		return res