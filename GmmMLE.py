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
from utils import *

class GmmMLE:

	GmmMLEParams = namedtuple('GmmMLEParams', ['color', 'mean', 'cov', 'covInverse', 'mixtureProbabilities'])

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

	def prob_x_cl_gmm(self, x, mean, covariance, covarianceInverse, mixtureProbabilities):
		result = 0
		for k in range(len(mean)):
			result = result + mixtureProbabilities[k] * self.prob_x_cl_gaussian(x, np.transpose(mean[k]), covariance[k], covarianceInverse[k])
		return result

	def prob_x_cl_gmm_gaussian(self, x, mean, covariance, covarianceInverse):
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

	def logSumExp(self, N, pi):
		summedArray = [(N[g] + math.log(pi[g])) for g in range(len(N))]
		maxElement = max(summedArray)

		result = maxElement + math.log(sum([math.exp(summedArray[g] - maxElement) for g in range(len(N))]))
		return result

	def gmmPredictHelperSingleGaussian(self, x, model):
		pass

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

	def predict(self, model, file):
		if len(model.color) == 1:
			img = cv2.imread(os.path.join(self.DATA_FOLDER, file))
			img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
			res = np.apply_along_axis(self.gmmPredictHelperSingleGaussian, 2, img, model)
		else:
			img = cv2.imread(os.path.join(self.DATA_FOLDER, file))
			img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
			res = np.apply_along_axis(self.gmmPredictHelperManyGaussians, 2, img, model)
		return res

	def gmmPredictLookupHelper(self, x, model):
		return model.item(x[0], x[1], x[2])

	def predictWithLookupTable(self, model, file):
		img = cv2.imread(os.path.join(self.DATA_FOLDER, file))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
		res = np.apply_along_axis(self.gmmPredictLookupHelper, 2, img, model)
		return res

	def getLookupTable(self, model):
		res = np.zeros((256, 256, 256), dtype=bool)
		x, y, z = res.shape

		for i in xrange(x):
			print i
			for j in xrange(y):
				for k in xrange(z):
					if len(model.color) == 1:
						res.itemset((i, j, k), self.gmmPredictHelperSingleGaussian(np.asarray([i, j, k]), model))
					else:
						res.itemset((i, j, k), self.gmmPredictHelperManyGaussians(np.asarray([i, j, k]), model))

		return res

	def initializeEMparameters(self, k):
		mu =  [255 * np.random.random_sample((1, 3)) for _ in range(k)]
		sigma = [70 * np.identity(3)] * k
		mixProb = [(1.0/k)] * k
		return mu, sigma, mixProb

	def EM(self, X):
		numTry = 2
		k = 3 # Number of mixtures
		n = X.shape[0]
		logLikelihood = 0; maxLikelihood = float("-inf")
		best_mu, best_sigma, best_mixProb = self.initializeEMparameters(k)

		for trial in xrange(numTry):
			print 'Trial: ' + str(trial)
			mu, sigma, mixProb = self.initializeEMparameters(k)
			sigmaInverse = [np.linalg.pinv(s) for s in sigma]
			membership = np.zeros((n, k))
			previousLL = 0
			numSaturation = 0

			while True:

				# Pg 438-439 PRML, Bishop
				# E-step = evaluate membership probabilities using current parameter values
				for i in xrange(membership.shape[0]):
					N_log = [self.prob_x_cl_gmm_gaussian((X[i, :]).reshape(3, 1), np.transpose(m), s, si) for m, s, si in zip(mu, sigma, sigmaInverse)]
					denominatorLog = self.logSumExp(N_log, mixProb) # sum(mixProb[g] * N[g] for g in range(len(N)))



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
				print 'Mixture Probabilities: ',
				print mixProb

				for j in xrange(k):
					# mu[j] = (1.0/Nk[j]) * np.average(X, axis=0, weights=membership[:, j])
					cumSum = np.zeros((1, 3))
					for i in range(n):
						cumSum = cumSum + membership[i, j] * X[i, :]
					mu[j] = (1.0/Nk[j]) * cumSum
				print 'Mu calculated: ',
				print mu

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
				# print sigma
				sigmaInverse = [np.linalg.pinv(s) for s in sigma]
				# M-step done
				#################################################################################

				# Evaluate the log-likelihood
				logLikelihood = 0
				for i in xrange(n):
					N = [self.prob_x_cl_gmm_gaussian((X[i, :]).reshape(3, 1), np.transpose(m), s, si) for m, s, si in zip(mu, sigma, sigmaInverse)]
					probSum = self.logSumExp(N, mixProb) #sum(mixProb[g] * N[g] for g in range(len(N)))
					# print probSum
					logLikelihood = logLikelihood + (probSum)
				# Evaluate done
				#################################################################################

				print 'Log-likelihood: ',
				print logLikelihood

				# Check for convergence; Break if converged
				if(abs(previousLL - logLikelihood) < 1 and numSaturation >= 1):
					break
				elif(abs(previousLL - logLikelihood) < 1 and numSaturation < 1):
					numSaturation = numSaturation + 1
				previousLL = logLikelihood
				# Check done

			if(maxLikelihood < logLikelihood):
				maxLikelihood = logLikelihood
				best_mu = mu
				best_sigma = sigma
				best_mixProb = mixProb

		return best_mu, best_sigma, best_mixProb, [np.linalg.pinv(s) for s in best_sigma]

	def train(self, training):
		mean = [None] * len(self.COLOR_LIST)
		covariance = [None] * len(self.COLOR_LIST)
		mixingProbabilites = [None] * len(self.COLOR_LIST)
		covarianceInverse = [None] * len(self.COLOR_LIST)

		for idx, color in enumerate(COLOR_LIST):
			roiPixels = np.empty((0,3), dtype=np.uint8)
			for file in training:
				img = cv2.imread(os.path.join(self.DATA_FOLDER, file))
				img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
				mask = getImageROIMask(file, color, self.DATA_FOLDER)
				roiPixelsInFile = getROIPixels(img, mask)
				roiPixels = np.concatenate([roiPixels, roiPixelsInFile])
			if roiPixels.shape[0] != 0:
				print 'Color: ' + str(color) 
				mu, sigma, mixProb, sigmaInverse = self.EM(roiPixels)
				mean[idx] = mu
				covariance[idx] = sigma
				mixingProbabilites[idx] = mixProb
				covarianceInverse[idx] = sigmaInverse

		model = self.GmmMLEParams(color=self.COLOR_LIST, mean=mean, cov=covariance, covInverse=covarianceInverse, mixtureProbabilities=mixingProbabilites)
		return model