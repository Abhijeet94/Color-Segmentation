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
from scipy.misc import logsumexp
from utils import *

class GmmMLE:

	def __init__(self, colorList, dataFolder, numMixtures=3, covMethod = 'FullCov'):
		self.COLOR_LIST = colorList
		self.DATA_FOLDER = dataFolder
		self.K = numMixtures
		self.covMethod = covMethod

	def prob_x_cl_gaussian(self, x, mean, covariance, covarianceInverse):
		x = x.reshape(3, 1)
		constant = (np.linalg.det(covarianceInverse) ** (1/2.0)) / ((((2 * math.pi) ** 3)) ** (1/2.0))
		exp1 = np.matmul(np.transpose(x - mean), covarianceInverse)
		exp2 = np.matmul(exp1, (x - mean))
		exponent = -0.5 * exp2
		result = constant * math.exp(exponent)
		return result

	def prob_x_cl_gmm(self, x, mean, covariance, covarianceInverse, mixtureProbabilities):
		result = 0
		for k in range(len(mean)):
			result = result + mixtureProbabilities[k] * self.prob_x_cl_gaussian(x, np.transpose(mean[k]), covariance[k], covarianceInverse[k])
		return result

	def log_prob_x_cl_gaussian(self, x, mean, covariance, covarianceInverse):
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

	def multivariateNormalLogPdf(self, X, mean, covariance, covarianceInverse):
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

	def gmmPredictHelperManyGaussians(self, x, model):
		# threshold = 1e-07 #for RGB
		threshold = 0#1e-06 #for Y_CR_CB

		red_barrel_probability = self.prob_x_cl_gmm(x, model.mean[0], model.cov[0], model.covInverse[0], model.mixtureProbabilities[0])

		max_other_probability = 0
		for c in range(1, len(model.color)):
			this_color_probability = self.prob_x_cl_gmm(x, model.mean[c], model.cov[c], model.covInverse[c], model.mixtureProbabilities[c])
			if this_color_probability > max_other_probability:
				max_other_probability = this_color_probability

		if red_barrel_probability > max_other_probability and red_barrel_probability > threshold:
			return True
		else:
			return False

	def predict(self, model, img):
		threshold = -110
		# img = cv2.imread(os.path.join(self.DATA_FOLDER, file))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

		bigMat = np.zeros((img.shape[0], img.shape[1], len(model.color)))
		for c, color in enumerate(model.color):
			k = len(model.mean[0])
			localBigMat = np.zeros((img.shape[0], img.shape[1], k))
			for j in range(k):
				# localBigMat[:, :, j] = math.log((model.mixtureProbabilities[c])[j]) +  multivariate_normal.logpdf(img, mean=(model.mean[c])[j].reshape(3), cov=(model.cov[c])[j])
				localBigMat[:, :, j] = math.log((model.mixtureProbabilities[c])[j]) +  self.multivariateNormalLogPdf(img, (model.mean[c])[j], (model.cov[c])[j], (model.covInverse[c])[j])
			bigMat[:, :, c] = logsumexp(localBigMat, axis=2).reshape(img.shape[0], img.shape[1])
		res = np.argmax(bigMat, axis = 2)
		res = res == 0

		resThreshold = np.amax(bigMat, axis = 2) > threshold
		res = np.logical_and(res, resThreshold)
		return res

	def gmmPredictLookupHelper(self, x, model):
		return model.item(x[0], x[1], x[2])

	def predictWithLookupTable(self, model, file):
		img = cv2.imread(os.path.join(self.DATA_FOLDER, file))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
		res = np.apply_along_axis(self.gmmPredictLookupHelper, 2, img, model)
		return res

	def getLookupTableSlow(self, model):
		res = np.zeros((256, 256, 256), dtype=bool)
		x, y, z = res.shape

		for i in xrange(x):
			print i
			for j in xrange(y):
				for k in xrange(z):
					res.itemset((i, j, k), self.gmmPredictHelperManyGaussians(np.asarray([i, j, k]), model))

		return res

	def getLookupTable(self, model):
		threshold = -110
		res = np.transpose(np.indices((256, 256, 256)), (1, 2, 3, 0))

		bigMat = np.zeros((res.shape[0], res.shape[1], res.shape[2], len(model.color)))
		for c, color in enumerate(model.color):
			k = len(model.mean[0])
			localBigMat = np.zeros((res.shape[0], res.shape[1], res.shape[2], k))
			for j in range(k):
				# localBigMat[:, :, :, j] = math.log((model.mixtureProbabilities[c])[j]) +  multivariate_normal.logpdf(res, mean=(model.mean[c])[j].reshape(3), cov=(model.cov[c])[j])
				localBigMat[:, :, :, j] = math.log((model.mixtureProbabilities[c])[j]) +  self.multivariateNormalLogPdf(res, (model.mean[c])[j], (model.cov[c])[j], (model.covInverse[c])[j])
			bigMat[:, :, :, c] = logsumexp(localBigMat, axis=3).reshape(res.shape[0], res.shape[1], res.shape[2])
		res = np.argmax(bigMat, axis = 3)
		res = res == 0

		resThreshold = np.amax(bigMat, axis = 3) > threshold
		res = np.logical_and(res, resThreshold)
		return res

	def initializeEMparameters(self, k):
		mu =  [255 * np.random.random_sample((1, 3)) for _ in range(k)]
		sigma = [70 * np.identity(3)] * k
		mixProb = [(1.0/k)] * k
		return mu, sigma, mixProb

	def EM(self, X, covMethod = 'FullCov'):
		numTry = 1
		k = self.K # Number of mixtures
		n = X.shape[0]
		logLikelihood = 0; maxLikelihood = float("-inf")
		best_mu, best_sigma, best_mixProb = self.initializeEMparameters(k)

		for trial in xrange(numTry):
			while True: # In case something breaks (like singular matrix)
				try:
					mu, sigma, mixProb = self.initializeEMparameters(k)
					sigmaInverse = [np.linalg.pinv(s) for s in sigma]
					membership = np.zeros((n, k))
					previousLL = 0
					numSaturation = 0

					while True: # EM iterations

						for j in range(k):
							# membership[:, j] = math.log(mixProb[j]) + multivariate_normal.logpdf(X, mean=mu[j].reshape(3), cov=sigma[j])
							membership[:, j] = math.log(mixProb[j]) + self.multivariateNormalLogPdf(X, mu[j], sigma[j], sigmaInverse[j])
						membership = np.exp(membership - logsumexp(membership, axis=1)[:,None])
						# E-step done
						#################################################################################

						# M-step = Re-estimate the parameters using current membership probabilities
						Nk = np.sum(membership, axis=0)
						mixProb = (1.0/n) * Nk

						for j in xrange(k):
							cumSum = np.sum(np.multiply(membership[:, j].reshape(n, 1), X), axis=0).reshape(1, 3)
							mu[j] = (1.0/Nk[j]) * cumSum

						if covMethod == 'FullCov':
							for j in xrange(k):
								shiftedX = np.subtract(X, mu[j])
								cumSum = np.zeros((3, 3))
								for it in (range((n/100000) + 1)):
									start = it * 100000
									end = min((it+1) * 100000, n)
									for r in range(3):
										for t in range(3):
											prod_r_t = np.multiply(shiftedX[start:end, r], shiftedX[start:end, t])
											cumSum[r,t] = cumSum[r,t] + np.sum(np.multiply(membership[start:end, j].reshape((end-start), 1), prod_r_t.reshape((end-start), 1)), axis=0)
								sigma[j] = (1.0/Nk[j]) * cumSum
						elif covMethod == 'DiagonalCov':
							for j in range(k):
								shiftedX = np.subtract(X, mu[j])
								shiftedSqX = np.square(shiftedX)
								sigma[j] = (1.0/Nk[j]) * np.diag(np.sum(np.multiply(membership[:, j], shiftedSqX), axis=0))

						sigmaInverse = [np.linalg.pinv(s) for s in sigma]
						# M-step done
						#################################################################################

						# Evaluate log-likelihood
						tempN = np.zeros((n, k))
						for j in range(k):
							# tempN[:, j] = math.log(mixProb[j]) + multivariate_normal.logpdf(X, mean=mu[j].reshape(3), cov=sigma[j])
							tempN[:, j] = math.log(mixProb[j]) + self.multivariateNormalLogPdf(X, mu[j], sigma[j], sigmaInverse[j])
						logLikelihood = np.sum(logsumexp(membership, axis=1).reshape(n, 1), axis = 0)
						# Evaluate done
						#################################################################################

						# Check for convergence; Break if converged
						if(abs(previousLL - logLikelihood) < 0.4 and numSaturation >= 1):
							break
						elif(abs(previousLL - logLikelihood) < 0.4 and numSaturation < 1):
							numSaturation = numSaturation + 1
						previousLL = logLikelihood
						# Check done
				except KeyboardInterrupt:
					exit(0)
				except Exception, e:
					print 'Broken instance: ' + str(e)
				else:
					break

			if(maxLikelihood < logLikelihood):
				maxLikelihood = logLikelihood
				best_mu = mu
				best_sigma = sigma
				best_mixProb = mixProb

		print best_mu
		return best_mu, best_sigma, best_mixProb, [np.linalg.pinv(s) for s in best_sigma]

	def train(self, training):
		mean = [None] * len(self.COLOR_LIST)
		covariance = [None] * len(self.COLOR_LIST)
		mixingProbabilites = [None] * len(self.COLOR_LIST)
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
				print '\nColor: ' + str(color) 
				mu, sigma, mixProb, sigmaInverse = self.EM(roiPixels, self.covMethod)
				mean[idx] = mu
				covariance[idx] = sigma
				mixingProbabilites[idx] = mixProb
				covarianceInverse[idx] = sigmaInverse

		model = GmmMLEParams(color=self.COLOR_LIST, mean=mean, cov=covariance, covInverse=covarianceInverse, mixtureProbabilities=mixingProbabilites)
		return model