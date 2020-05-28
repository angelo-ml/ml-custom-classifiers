################################################
## Python Machine Learning custom classifiers ##
################################################
## Author: Malandrakis Angelos
## 28/05/2020
################################################


# load external libraries
import pandas as pd
import numpy as np
import math
import scipy.cluster as sc
import matplotlib.pyplot as plt
from random import shuffle



##############################################
## ==== Logistic Regression Classifier ==== ##
##############################################

class LogisticRegression:
	
	def __init__(self, alpha):
		self.alpha = alpha
		
	def fit(self, x, y):
		# standardize data and convert them to numpy array
		x = self._dataStandardization(x)
		
		# Initialize data
		self.beta = [1] * ((x.shape[1]) + 1)

		for i in range(len(x)):
			
			self.beta[0] = self.beta[0] - self.alpha * ( self._sigma(x[i]) - y[i] ) * self._sigma(x[i]) * (1-self._sigma(x[i])) * 1

			for j in range(1,len(self.beta)):
				self.beta[j] = self.beta[j] - self.alpha * ( self._sigma(x[i]) - y[i] ) * self._sigma(x[i]) * (1-self._sigma(x[i])) * x[i,j-1]
	
	def _dataStandardization(self, x):
		
		standardData = x.copy()
		rows = x.shape[0]
		cols = x.shape[1]
		self.sigma = [0] * x.shape[1]
		self.mu = [0] * x.shape[1]

		for j in range ( cols ):
			self.sigma[j] = np.std( x[:,j])
			self.mu[j] = np.mean( x[:,j])
			for i in range ( rows ):
				standardData[i,j] = ( x[i,j] - self.mu[j])/ self.sigma[j]

		standardData = np.array(standardData)
		# shuffle data
		shuffle(standardData)
		return(standardData)

	def _testDataStandardization(self, x):
		
		standardData = x.copy()
		rows = x.shape[0]
		cols = x.shape[1]

		for j in range ( cols ):
			for i in range ( rows ):
				standardData[i,j] = ( x[i,j] - self.mu[j])/ self.sigma[j]

		standardData = np.array(standardData)
		# shuffle data
		shuffle(standardData)
		return(standardData)

	def _sigma(self, x):
			
		x = np.array([np.concatenate(([1],x))])
		t = np.matmul(self.beta,x.T)
		
		s = 1/(1+math.exp(-t))
		return(s)

	def predict(self, data):
		data = self._testDataStandardization(data)
		onesVector = np.array([[1] * data.shape[0]]).T
		data = np.hstack((onesVector,data))

		results = [0] * data.shape[0]
		for i in range(data.shape[0]):
			results[i] = round(self._sigma(data[i,:3]))	 

	def resultPrinter(self):
		beta = self.beta
		out = "result: " + str(beta[0]) + " + x1*" + str(beta[1]) + " + x2*" + str(beta[2])
		return(out)

	def score(self, X, y):
		testData = self._testDataStandardization(X)
		accuracyIndex = 0

		for i in range(testData.shape[0]):
			result = round(self._sigma(testData[i,:]))

			if result == y[i]:
				accuracyIndex = accuracyIndex + 1
		
		# compute accuracy
		accuracy = accuracyIndex/testData.shape[0]
		return(accuracy)



		
##############################################
## ======= Decision Tree Classifier ======= ##
##############################################

# create the node class which will be used to build the decision tree
class Node:
	def __init__(self):
		self.predicted_class = None
		self.splitting_attribute = None
		self.splitting_point = None
		self.left = None
		self.right = None
	

class DecisionTree:
	def __init__(self, max_depth = None, min_samples_split=2):
		
		# max depth is the maximum depth of the tree path from the root node to the leafs
		self.max_depth = max_depth	
		
		# min_samples_split is the minimum number of data required to split a node.
		# default value is 2
		self.min_samples_split = min_samples_split
		
			
	def fit(self, X, y):
		
		# stores the classes of the dataset
		classes = []
		for label in y:
			if label not in classes:
				classes.append(label)
		
		if (self.max_depth is None):
			# this is the absolute maximum depth. 
			# The classifier should never reach this
			self.max_depth = len(X)-1
		
		self.classes = classes
		self.num_classes = len(classes)
		self.num_features = X.shape[1]
		
		# builds the decision tree recursively
		self.decisionTree = self._decisionTreeBuilder(X, y)
		
		return self
		
	# finds the best split of the dataset, based on the gini index
	def _node_splitting(self, X, y):
		
		m = len(y)
		
		# Count the data of each class in the current node.
		node_classes = []
		for label in y:
			if label not in node_classes:
				node_classes.append(label)
				
		node_classes_counts = [0] * self.num_classes
		for i in range(self.num_classes):
			node_classes_counts[i] = sum(y == self.classes[i])

		# initialize gini index with the maximum value
		best_gini_value = 1
		
		# Loop through all features and all the data to find the best Ginin Index
		for feature_index in range(self.num_features):
			
			# Sort data for the selected feature
			data = np.vstack((X[:, feature_index], y)).T
			sorted_data = data[data[:,0].argsort()]
			labels = sorted_data[:,1].astype(int)
			feature_datapoints = sorted_data[:,0]
			
			# counts the classes of the data for the two splits of the node
			classes_left = np.array([0] * self.num_classes)			
			classes_right = np.array(node_classes_counts)
			
			# Loop through all the data of the node
			for i in range(1, m):
				
				previous_label = labels[i - 1]
				
				previous_label_id = self.classes.index(previous_label)
				classes_left[previous_label_id] = classes_left[previous_label_id] + 1
				classes_right[previous_label_id] = classes_right[previous_label_id] - 1
				
				# compute gini index for the two spits
				gini_left = 1.0 - sum( (classes_left/i)**2 )
				gini_right = 1.0 - sum( (classes_right/(m-i))**2 )
				
				
				### ==== This was the original part of the code that added too much complexity ==== ##
				# p_left = 0
				# p_right = 0
				# for node_class in self.classes:
				#	 left_data = np.where(feature_datapoints < i)
				#	 p_left = p_left + (len(np.where(labels[left_data] == node_class)) / len(left_data)) **2
					
				#	 right_data = np.where(feature_datapoints >= i)
				#	 p_right = p_right + (len(np.where(labels[right_data] == node_class)) / len(right_data)) **2
				
				
				# gini_left = 1.0 - p_left
				# gini_right = 1.0 - p_right

				
				# Computes GiniA based on the gini indexes of the two splits
				gini = (i/m) * gini_left + ((m - i)/ m) * gini_right

				# If the new GiniA is smaller than the new one, 
				# the best gini index, best splitting attribute and best splitting point 
				# are updated
				if gini < best_gini_value:
					best_gini_value = gini
					best_splitting_attribute = feature_index
					best_splitting_point = (feature_datapoints[i] + feature_datapoints[i - 1]) / 2  # midpoint

		return [best_splitting_attribute, best_splitting_point] 

	
	# the method that builds the decision tree recursively
	def _decisionTreeBuilder(self, X, y, depth=0):
		
		# create a tree node each time it's called
		node = Node()

		# Split recursively until maximum depth is reached 
		# or until the node has less than the minimum required data to be split
		if depth < self.max_depth and len(y) >= self.min_samples_split:
			
			# get the splitting criterion (splitting-point and splitting-attribute)
			# based on the Gini Index
			splitting_attribute = self._node_splitting(X, y)[0]
			splitting_point = self._node_splitting(X, y)[1]
			
			# split the data of the current node 
			indices_left = np.where(X[:, splitting_attribute] < splitting_point)
			indices_right = np.where(X[:, splitting_attribute] >= splitting_point)
			X_left = X[indices_left]
			y_left = y[indices_left]
			X_right = X[indices_right]
			y_right = y[indices_right]
			
			# update the splitting-criterion of the current node
			node.splitting_attribute = splitting_attribute
			node.splitting_point = splitting_point
			
			# create the left and right nodes recursively
			node.left = self._decisionTreeBuilder(X_left, y_left, depth + 1)
			node.right = self._decisionTreeBuilder(X_right, y_right, depth + 1)
		
		# If the node cannot be split any more we assign it a label of the majority class
		# to make it a leaf node
		else:
			
			# find the mojority class
			data_per_class = [0] * self.num_classes
			for i in range(self.num_classes):
				data_per_class[i] = np.sum(y == i)
			
			# assign the majority class label to the node
			predicted_class = np.argmax(data_per_class)
			node.predicted_class = predicted_class
		
		return node
	
	def predict(self, X):
		
		predictions = np.zeros(X.shape[0])
		
		# loop through the data to predict the label for each one of the data-points
		for i in range(X.shape[0]):
			
			# check the spiting criteria of each node beginning from the root node
			# until we reach a node that doesn't have any children nodes
			node = self.decisionTree
			while node.left:
				if X[i,node.splitting_attribute] < node.splitting_point:
					node = node.left
				else:
					node = node.right
			
			# we assign to the data-point the label of the leaf node
			predictions[i] = node.predicted_class
		
		return(predictions)
		
	
	# return the accuracy of the classifier given a test-set and its labels
	def score(self, X, y):
		y_predict = self.predict(X)
		
		false_predictions = 0
		for i in range(len(y_predict)):
			if y_predict[i] != y[i]:
				false_predictions = false_predictions + 1
		
		
		return 1-( false_predictions/len(y) )
