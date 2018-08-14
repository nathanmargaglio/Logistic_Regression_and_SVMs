import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from random import randrange
from sklearn.svm import SVC
import logging
import time

# set up logging to file - see previous section for more details
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='log_' + str(int(time.time())) + '.txt',
                    filemode='w')
# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

def preprocess():
    """
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]

    intercept = np.ones((n_data, 1))
    x = np.hstack((intercept, train_data))
    h = sigmoid(np.dot(x, initialWeights)).reshape(n_data, 1)
    y = labeli

    error = -sum(y * np.log(h) + (1 - y) * np.log(1 - h)) / n_data
    error_grad = np.dot(x.T, (h - y)).reshape(n_features + 1, ) / n_data

    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """

    n_data = data.shape[0]

    intercept = np.ones((n_data, 1))
    x = np.hstack((intercept, data))

    label = sigmoid(np.dot(x, W))
    _label = np.argmax(label, axis=1)

    return _label.reshape((_label.shape[0],1))


def softmax(y_linear):
    exp = np.exp(y_linear-np.max(y_linear, axis=1).reshape((-1,1)))
    norms = np.sum(exp, axis=1).reshape((-1,1))
    return exp / norms

def mlrObjFunction(initialWeights, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, Y = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]

    intercept = np.ones((n_data, 1))
    x = np.hstack((intercept, train_data))
    h = softmax(np.dot(x, initialWeights.reshape(n_features + 1, 10))).reshape(n_data, 10)
    y = Y

    error = -sum(sum(y * np.log(h)))
    error_grad = np.dot(x.T, (h - y)).reshape((n_features + 1)*10, )

    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    n_data = data.shape[0]

    intercept = np.ones((n_data, 1))
    x = np.hstack((intercept, data))

    return _label.reshape((_label.shape[0],1))


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
logging.info('Starting Logistic Regression')
for i in range(n_class):
    logging.info('Class:' + str(i))
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# dump to pickle
W.dump('params.pickle')

predicted_label = blrPredict(W, train_data)
logging.info('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
logging.info('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
logging.info('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label))) + '%')

"""
Script for Support Vector Machine
"""

logging.info('\n\n--------------SVM-------------------\n\n')

logging.info("\nLinear")
clf = SVC(kernel='linear')
clf.fit(train_data, train_label)

predicted_label = clf.predict(train_data)
logging.info('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = clf.predict(validation_data)
logging.info('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = clf.predict(test_data)
logging.info('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label))) + '%')


logging.info("\nRadial; Gamma: 1")
clf = SVC(kernel='rbf', gamma=1)
clf.fit(train_data, train_label)

predicted_label = clf.predict(train_data)
logging.info('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = clf.predict(validation_data)
logging.info('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = clf.predict(test_data)
logging.info('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label))) + '%')


logging.info("\nRadial; Gamma: Default")
clf = SVC(kernel='rbf')
clf.fit(train_data, train_label)

predicted_label = clf.predict(train_data)
logging.info('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = clf.predict(validation_data)
logging.info('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = clf.predict(test_data)
logging.info('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label))) + '%')


preds = {}

for C in [1] + list(np.arange(10, 110, 10)):
    logging.info("\nRadial; C:", C)
    clf = SVC(C=C)
    clf.fit(train_data, train_label)

    preds[C] = {}

    predicted_label = clf.predict(train_data)
    logging.info('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label))) + '%')
    preds[C]['train'] = 100 * np.mean((predicted_label == train_label))

    # Find the accuracy on Validation Dataset
    predicted_label = clf.predict(validation_data)
    logging.info('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label))) + '%')
    preds[C]['validation'] = 100 * np.mean((predicted_label == validation_label))

    # Find the accuracy on Testing Dataset
    predicted_label = clf.predict(test_data)
    logging.info('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label))) + '%')
    preds[C]['testing'] = 100 * np.mean((predicted_label == test_label))


"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
logging.info('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
logging.info('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
logging.info('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
