'''
file: helperFunctions.py
Author:	Roberto Vega
Email:	rvega_at_ualberta_dot_ca
Description:
	This file contains a helper function to load the data that will be used on the experiments
	for the PAC 2018 competition.
'''
import sys

# Check if using python 2 or 3
if sys.version_info.major == 2:
	import cPickle as pickle
else:
	import pickle

def get_experiment_data(exp_num):
	'''
	This function loads a precomputed file: 'Experiment_data.pkl' that contains the subject that will
	be used for training/testing purposes. We have data for 10 experiments. The data in the pkl contains
	70% of the data for trainig purposes and the remaining 30% for testing purposes. The data was divided
	in a stratified way --i.e. the proportion of positive samples is the same in both, training and test set.

	Inputs:
		- exp_num: it is a number between 0 - 9 indicating the experiment ID
	Outputs:
		- train_info: A matix of dimensions 1254 x 5. The first column contains a string with the subject_ID.
		The rest of the columns are: label, age, gender and TIV
		- test_info: Similar to train_info, but for testing purposes. The size of this matrix is 538 x 5

	Example:

		train_info, test_info = get_experiment_data(0)
		print(train_info[0, :])
		  ['PAC2018_0592' 2 51 2 1658.0]
		print(test_info[0, :])
		  ['PAC2018_1807' 1 27 2 1620.0]
	'''

	# Load the pickle files with the data to use as train and test sets in a given experiment
	matrices = pickle.load(open('Experiment_data.pkl', 'rb'))
	
	# Extract the train and test matrices
	train_matrices = matrices[0]
	test_matrices = matrices[1]

	train_info = train_matrices[exp_num]
	test_info = test_matrices[exp_num]
	
	return train_info, test_info