import numpy as np
import matplotlib.pyplot as plt
import os

def shuffle_list_array_inplace(A, l):
    # Transform list to array
    L = np.vstack(l)
    # Reshape for future broadcast
    L = L.reshape(L.shape[0], )

    # Concatenate Array an broadcasted list
    X = np.concatenate((np.broadcast_to(np.array(L)[:, None, None], A.shape[:-1] + (1,)), A), axis=-1)

    # Shuffle many time (just to be sure)
    for i in range(30):
        np.random.shuffle(X)

    return X[:, :, 1:], X[:, 0, 0]

def load_data() :
    # Images paths
    train_sylveon = '../data/test_set/sylveon/'
    test_sylveon = '../data/test_set/sylveon/'
    train_pikachu = '../data/train_set/pikachu/'
    test_pikachu = '../data/test_set/pikachu/'

    # Test set
    # Loading images in rank-3 tensor
    X_train_pika = np.array([np.array(np.dot(plt.imread(train_pikachu + file)[..., :3], [0.2989, 0.5870, 0.1140]))
                             for file in os.listdir(train_pikachu)])
    X_train_sylveon = np.array([np.array(np.dot(plt.imread(train_sylveon + file)[..., :3], [0.2989, 0.5870, 0.1140]))
                                for file in os.listdir(train_sylveon)])

    # Concatenate the two tensor of images
    X_train = np.concatenate((X_train_sylveon, X_train_pika), axis=0)
    y_train = [0 for i in range(X_train_sylveon.shape[0])] + [1 for i in range(X_train_pika.shape[0])]

    #shuffle the training set
    X_train, y_train = shuffle_list_array_inplace(X_train, y_train)

    # Trest set
    # Loading images in rank-3 tensor
    X_test_pika = np.array([np.array(np.dot(plt.imread(test_pikachu + file)[..., :3], [0.2989, 0.5870, 0.1140]))
                             for file in os.listdir(test_pikachu)])
    X_test_sylveon = np.array([np.array(np.dot(plt.imread(test_sylveon + file)[..., :3], [0.2989, 0.5870, 0.1140]))
                                for file in os.listdir(test_sylveon)])

    # Concatenate the two tensor of images
    X_test = np.concatenate((X_test_sylveon, X_test_pika), axis=0)
    y_test = [0 for i in range(X_test_sylveon.shape[0])] + [1 for i in range(X_test_pika.shape[0])]

    # shuffle the training set
    X_test, y_test = shuffle_list_array_inplace(X_test, y_test)

    return X_train, y_train, X_test, y_test
