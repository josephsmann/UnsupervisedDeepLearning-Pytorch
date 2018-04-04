'''
file: helperFunctions.py
Author:    Roberto Vega
Email:    rvega_at_ualberta_dot_ca
Description:
    This file contains a helper function to load the data that will be used on the experiments
    for the PAC 2018 competition.
'''
import sys
import nibabel as nib
import numpy as np
from nilearn.image import resample_img


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
    '''
    
    # Load the pickle files with the data to use as train and test sets in a given experiment
    
    # JM: using 'with' ensure file is closed afterward
    with open('Experiment_data_wscanner.pkl', 'rb') as f:
        matrices = pickle.load(f)
    
    # Extract the train and test matrices
    train_matrices = matrices[0]
    test_matrices = matrices[1]

    train_info = train_matrices[exp_num]
    test_info = test_matrices[exp_num]
    
    return train_info, test_info

def extract_vector_features_from_matrix(matrix, datapath):
    '''
    This function will transform a matrix obtained by the function get_experiment_data into a feature
    matrix and label vector.

    Inputs:
        - matrix: one of the matrices created with get_experiment data. It is a matrix of dimensions
            num_subjetcs x 5, where the first column contains the Subject_ID, the second contains the label.
    Outputs:
        - X: a matrix of num_subjects x num_features. The number of features are all the voxels that are
            in a mask computed previousl + the 3 covariates (Age, Gender, TIV)
        - y: A vector of length num_subjects that contains the labels for every row in X.
    '''

    # Start by loading the mask
    mask = pickle.load(open('mask.pkl', 'rb'))

    # Extract the number of subject in the matrix
    num_subjects = matrix.shape[0]
    num_features = np.sum(mask)

    X = np.zeros((num_subjects, num_features))
    y = matrix[:, 1]

    # Extract the labels

    for i in range(num_subjects):
        # Extract the subject ID and concatenate it the extension and datapath
        subject_ID = matrix[i, 0]
        filename = datapath + subject_ID + '.nii'

        # Load the nii file
        img = nib.load(filename)
        
        # Extract the 3D data and extend it into a single vector
        data = img.get_data()
        data_vector = np.reshape(data, (-1))
        actual_data = data_vector[mask]

        X[i, 0:num_features] = actual_data

    return X, np.int8(y)

def extract_covariates_from_matrix(matrix):
    from sklearn.preprocessing import OneHotEncoder

    gender_encoder = OneHotEncoder(sparse=False)
    hospital_encoder = OneHotEncoder(sparse=False)

    # Extract the labels
    y = np.int8(matrix[:, 1])

    # Extract the covariates
    age = np.reshape(matrix[:, 2], (-1,1))
    gender = np.reshape(matrix[:, 3], (-1,1))
    TIV = np.reshape(matrix[:, 4], (-1,1))
    scanner = np.reshape(matrix[:, 5], (-1,1))

    # Use one hot encoding on the gender and scanner
    gender = gender_encoder.fit_transform(gender)
    scanner = hospital_encoder.fit_transform(scanner)

    # Concatenate all the features into a single table
    X = np.hstack([age, gender, TIV, scanner])
    # X = np.hstack([age, gender, TIV])


    return X, y

def extract_3D_from_matrix(matrix, datapath):
    '''
    This function will transform a matrix obtained by the function get_experiment_data into a feature
    matrix and label vector.

    Inputs:
        - matrix: one of the matrices created with get_experiment data. It is a matrix of dimensions
            num_subjetcs x 5, where the first column contains the Subject_ID, the second contains the label.
    Outputs:
        - X: a matrix of num_subjects x num_features. The number of features are all the voxels that are
            in a mask computed previousl + the 3 covariates (Age, Gender, TIV)
        - y: A vector of length num_subjects that contains the labels for every row in X.
    '''

    # Extract the bounding box contaiing the limits that actually contain data
    bounding_box = pickle.load(open('bounding_box.pkl', 'rb'))
    dim_0 = bounding_box[0]
    dim_1 = bounding_box[1]
    dim_2 = bounding_box[2]

    # Extract the number of subject in the matrix
    num_subjects = matrix.shape[0]

    X = np.ones([num_subjects, 95, 125, 100, 1])
    y = matrix[:, 1]

    # Extract the data from every image and store it in a 5-D array
    for i in range(num_subjects):
        # Extract the subject ID and concatenate it the extension and datapath
        subject_ID = matrix[i, 0]
        filename = datapath + subject_ID + '.nii'

        # Load the nii file
        img = nib.load(filename)
        
        # Extract the 3D data and extend it into a single vector
        data = img.get_data()
        data = data[dim_0[0]:dim_0[1], dim_1[0]:dim_1[1], dim_2[0]:dim_2[1]]
        X[i,:,:,:,0] = data

    return X, np.int8(y)

if __name__ == "__main__":
    print(-1)
