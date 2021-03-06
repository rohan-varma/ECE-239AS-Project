import numpy as np
import math as math
import h5py
import sys
from sklearn.model_selection import train_test_split

class EEGDataLoader(object):
	def __init__(self, data_path):
		self.data_path = data_path #'drive/239 Project/project/project_datasets/' or 'project_datasets/'

	def load_all_data(self):
		"""Loads all the data from the EEG dataset.
		The testing data are made by randomly sampling 50 points from each of the 9 datasets, and then concatenating them together
		returns 4-tuple of:
			X_train: np.array of shape (9, 238, 25, 1000), data features
			y_train: np.array of shape (9, 238), data labels
			X_test: np.array of shape (9, 50, 25, 1000), testing features
			y_test np.array of shpe (9, 50), testing labels
		"""
		np.random.seed(239)
		dataset_path = self.data_path
		# dataset_path = 'project_datasets/' for not google drive
		data_files = [dataset_path + 'A0{}T_slice.mat'.format(i) for i in range(1, 10)]
		X_train, y_train, X_test, y_test = [], [], [], []
		# go through each file and parse the data
		for file in data_files:
			A0T = h5py.File(file, 'r')
			X = np.copy(A0T['image'])
			X = np.clip(X, a_min = np.finfo(float).eps, a_max = None)
			y = np.copy(A0T['type'])
			y = y[0,0:X.shape[0]:1]
			y = np.asarray(y, dtype=np.int32)
			X = X[:, :-3]
			good_trials_X, good_trials_y = [], []
			# go through the trials
			for i in range(X.shape[0]):
				if np.any(np.isnan(X[i])):
					print('removing a nan entry')
					continue
				else:
					X[i] = (X[i] - np.mean(X[i], axis = 0)) /(np.std(X[i], axis = 0) + np.finfo(float).eps)
					good_trials_X.append(X[i])
					good_trials_y.append(y[i])
			X, y = np.array(good_trials_X), np.array(good_trials_y)
			# x_org = X.shape[0]
			# nan_trials = []
			# for i in range(X.shape[0]):
 		# 		for j in range(X.shape[1]):
   #     					if math.isnan(X[i,j,0]):
   #          					nan_trials.append(i) 
			# X = np.delete(X,np.asarray(nan_trials),axis = 0)
			# X = X[X != NaN]
			# y = np.copy(A0T['type'])
			# y = y[0,0:x_org:1]
			# y = np.asarray(y, dtype=np.int32)
			# y = np.delete(y, np.asarray(nan_trials))
			# X = X[:, :-3]
			# generate a list of 50 non-repeating indices in the range [0, 288)
			random_indices = set(np.random.choice(X.shape[0], 50, replace=False))
			cur_x, cur_y, cur_x_test, cur_y_test = [], [], [], []
			for i in range(X.shape[0]):
				# if the index is not in the random indices, it's a training point, else its a testing point.
				if i in random_indices:
					# test point
					cur_x_test.append(X[i])
					cur_y_test.append(y[i])
				else:
					# training point
					cur_x.append(X[i])
					cur_y.append(y[i])
			# convert everything to a np array
			cur_x, cur_y, cur_x_test, cur_y_test = np.array(cur_x), np.array(cur_y), np.array(cur_x_test), np.array(cur_y_test)
			X_train.append(cur_x)
			y_train.append(cur_y)
			X_test.append(cur_x_test)
			y_test.append(cur_y_test)
		# convert result to np array
		X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
		self.X_train, self.y_train, self.X_test, self.y_test = X_train, y_train, X_test, y_test
		print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
		return X_train, y_train, X_test, y_test


	def get_train_validation_splits(self, validation_p = 0.2):
		"""Divides the training data into training and validation splits
		Params:
		validation_p: proportion of data that should be held out for validation
		returns 4-tuple:
			X_train, y_train, X_val, y_val
		"""
		X_train, y_train, X_val, y_val = [], [], [], []
		for i in range(self.X_train.shape[0]):
			cur_x, cur_y = self.X_train[i], self.y_train[i]
			x, x_val, y, yv = train_test_split(cur_x, cur_y, test_size = validation_p, random_state = 239)
			X_train.append(x)
			y_train.append(y)
			X_val.append(x_val)
			y_val.append(yv)
		return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val)



	def get_ith_dataset(self, i, train = True):
		"""Gets the data corresponding to file i for 0<=i<=9 in the data files
		if train = True, it returns the training data, if train = False, it returns the testing data
		return (x, y) where x, y are the data for the ith file.
		"""
		return (self.X_train[i], self.y_train[i]) if train else (self.X_test[i], self.y_test[i])
		


if __name__ == '__main__':
	print(sys.argv)
	if len(sys.argv) != 2:
		print('USAGE: python3 load_data.py PATH')
		exit(1)
	else:
		data_path = sys.argv[1]
	data_loader = EEGDataLoader(data_path)
	X_train, y_train, x_test, y_test = data_loader.load_all_data()
	print('THE GOOD GOOD SHAPES')
	print(X_train.shape, y_train.shape, x_test.shape, y_test.shape)
	print('THATS ALL FOLKS')
	x_train, y_train, x_val, y_val = data_loader.get_train_validation_splits()
	print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
	x, y = data_loader.get_ith_dataset(3)
	print(x.shape, y.shape)
	x, y = data_loader.get_ith_dataset(3, train = False)
	print(x.shape, y.shape)

