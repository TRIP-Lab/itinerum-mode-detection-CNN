import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from sklearn.model_selection import train_test_split

import time

import pandas as pd

def split_train_test(X_origin, Y_orig):



    X_train_orig, X_test_orig, Y_train_orig, Y_test_orig = train_test_split(X_origin, Y_orig, test_size = 0.20, random_state = None)
    # print('shapes Y origin', Y_orig.shape)
    # indices = np.random.permutation(X_origin.shape[0])
    # print(' indices[80:] shape is:',  indices[80:].shape)
    # training_idx, test_idx = indices[80:], indices[:80]
    # X_train_orig, X_test_orig = X_origin[training_idx, :,:,:], X_origin[test_idx, :, :, :]
    # Y_train_orig, Y_train_orig = Y_orig[training_idx], Y_orig[test_idx]

    # training_idx = np.random.rand(X_origin.shape[0]) < 0.7
    # X_train_orig = X_origin[training_idx,:,:,: ]
    # X_test_orig = X_origin[~training_idx,:,:,:]
    # Y_train_orig = Y_orig[training_idx]
    # Y_test_orig = Y_orig[~training_idx]
    # print("shape is : ",X_train_orig.shape, X_test_orig.shape, Y_train_orig.shape, Y_test_orig.shape)
    # print("shape is : ", Y_test_orig)
    #np.save("CNN_labels_data_4channels_restericted2", Y_orig)
    #np.save("CNN_segmented_data_4channels_restericted2", X_orig)


    return (X_train_orig, X_test_orig, Y_train_orig, Y_test_orig)

def resampled_db(X_train, Y_train, X_test, Y_test, Y_train_labels, Y_test_labels, num_channels):
    #find the number of samples for each class in the dataset
    # decode the onehot array of Whole datset labels
    # decoded_one_hot = np.argmax(Y_train, axis=1)
    # decoded_one_hot_test = np.argmax(Y_test, axis=1)
    # Creating empty list and np array for saving train data
    sub_samples_train = []
    sub_samples_test = []
    # find the number of occurrence for each class
    Y_train_labels = np.rec.array(Y_train_labels)
    classes, counts = np.unique(Y_train_labels.class_label, return_counts=True)
    class_counter = dict(zip(classes, counts))

    Y_test_labels = np.rec.array(Y_test_labels)
    classes_test, counts_test = np.unique(Y_test_labels.class_label, return_counts=True)
    class_counter_test = dict(zip(classes_test, counts_test))

    #define numpy structured array ('https://docs.scipy.org/doc/numpy-1.10.0/user/basics.rec.html')

    X_train_car = np.zeros(class_counter[3], dtype=[('uuid', 'S64'), ('trip_id', 'int8'), ('segment_id', 'int8'), ('channels', '(70,6)float64')])
    X_train_car = np.rec.array(X_train_car)
    X_test_car = np.zeros(class_counter_test[3], dtype=[('uuid', 'S64'), ('trip_id', 'int8'), ('segment_id', 'int8'), ('channels', '(70,6)float64')])
    X_test_car = np.rec.array(X_test_car)

    non_cars_counts = sum([counts for classes, counts in class_counter.items() if classes != 3])
    non_cars_counts_test = sum([counts for classes, counts in class_counter_test.items() if classes != 3])

    X_train_others = np.zeros(non_cars_counts, dtype=[('uuid', 'S64'), ('trip_id', 'int8'), ('segment_id', 'int8'), ('channels', '(70,6)float64')])
    X_train_others = np.rec.array(X_train_others)
    X_test_others = np.zeros(non_cars_counts_test, dtype=[('uuid', 'S64'), ('trip_id', 'int8'), ('segment_id', 'int8'), ('channels', '(70,6)float64')])
    X_test_others = np.rec.array(X_test_others)

    # X_train_others = np.ndarray(shape=(non_cars_counts,70,num_channels), dtype = float)
    # X_test_others = np.ndarray(shape=(non_cars_counts_test, 70, num_channels), dtype=float)

    Y_train_car = np.zeros(class_counter[3], dtype=[('uuid', 'S64'), ('trip_id', 'int8'), ('segment_id', 'int8'), ('class_label', '(4,)int8')])
    Y_train_car = np.rec.array(Y_train_car)
    Y_test_car = np.zeros(class_counter_test[3], dtype=[('uuid', 'S64'), ('trip_id', 'int8'), ('segment_id', 'int8'), ('class_label', '(4,)int8')])
    Y_test_car = np.rec.array(Y_test_car)

    Y_train_others = np.zeros(non_cars_counts, dtype=[('uuid', 'S64'), ('trip_id', 'int8'), ('segment_id', 'int8'), ('class_label', '(4,)int8')])
    Y_train_others = np.rec.array(Y_train_others)
    Y_test_others = np.zeros(non_cars_counts_test, dtype=[('uuid', 'S64'), ('trip_id', 'int8'), ('segment_id', 'int8'), ('class_label', '(4,)int8')])
    Y_test_others = np.rec.array(Y_test_others)

    # Y_train_others = np.ndarray(shape=(non_cars_counts,num_channels), dtype = float)
    # Y_test_others = np.ndarray(shape=(non_cars_counts_test, num_channels), dtype=float)


    # X_train_car = np.ndarray(shape=(class_counter[3],70,num_channels), dtype = float)
    # X_test_car = np.ndarray(shape=(class_counter_test[3], 70, num_channels), dtype=float)
    #
    # non_cars_counts = sum([counts for classes, counts in class_counter.items() if classes != 3])
    # non_cars_counts_test = sum([counts for classes, counts in class_counter_test.items() if classes != 3])
    #
    # X_train_others = np.ndarray(shape=(non_cars_counts,70,num_channels), dtype = float)
    # X_test_others = np.ndarray(shape=(non_cars_counts_test, 70, num_channels), dtype=float)
    #
    # Y_train_car = np.ndarray(shape=(class_counter[3],num_channels), dtype = float)
    # Y_test_car = np.ndarray(shape=(class_counter_test[3],num_channels), dtype = float)
    #
    # Y_train_others = np.ndarray(shape=(non_cars_counts,num_channels), dtype = float)
    # Y_test_others = np.ndarray(shape=(non_cars_counts_test, num_channels), dtype=float)



    #find the averge number of samples per minor classes: bike,walk and transit
    avg_num_smpl_minor_classes = int((class_counter[0] + class_counter[1] + class_counter[2])/3)
    avg_num_smpl_minor_classes_test = int((class_counter_test[0] + class_counter_test[1] + class_counter_test[2])/3)


    #create seprate dbs for car and non-car samples
    counter_train_car = 0
    counter_train_others = 0
    for index, i in enumerate(Y_train_labels.class_label):

        if i == 3:
            # print(index)
            # print(X_train_car[index, :])
            X_train_car[counter_train_car] = X_train[index]
            Y_train_car[counter_train_car] = Y_train[index]
            counter_train_car += 1

            # for z in range(X_train_car.shape[0]):
            # print(X_train_car[index, :])
        else:
            X_train_others[counter_train_others] = X_train[index]
            Y_train_others[counter_train_others] = Y_train[index]
            counter_train_others += 1

    counter_test_car = 0
    counter_test_others = 0
    for index, i in enumerate(Y_test_labels.class_label):
        if i == 3:
            X_test_car[counter_test_car] = X_test[index]
            Y_test_car[counter_test_car] = Y_test[index]
            counter_test_car += 1
        else:
            X_test_others[counter_test_others] = X_test[index]
            Y_test_others[counter_test_others] = Y_test[index]
            counter_test_others += 1
    # for index, i in enumerate(decoded_one_hot):
    #
    #     if i == 3:
    #         # print(index)
    #         # print(X_train_car[index, :])
    #         X_train_car[counter_train_car,:,:] = X_train[index,:,:]
    #         Y_train_car[counter_train_car,:] = Y_train[index,:]
    #         counter_train_car += 1
    #
    #         # for z in range(X_train_car.shape[0]):
    #         # print(X_train_car[index, :])
    #     else:
    #         X_train_others[counter_train_others, :, :] = X_train[index, :, :]
    #         Y_train_others[counter_train_others, :] = Y_train[index, :]
    #         counter_train_others += 1
    #
    # counter_test_car = 0
    # counter_test_others = 0
    # for index, i in enumerate(decoded_one_hot_test):
    #     if i == 3:
    #         X_test_car[counter_test_car,:,:] = X_test[index,:,:]
    #         Y_test_car[counter_test_car,:] = Y_test[index,:]
    #         counter_test_car += 1
    #     else:
    #         X_test_others[counter_test_others, :, :] = X_test[index, :, :]
    #         Y_test_others[counter_test_others, :] = Y_test[index, :]
    #         counter_test_others += 1

    size_car_samples_test = class_counter_test[3]
    size_car_samples = class_counter_test[3]



    permutation = list(np.random.permutation(size_car_samples))
    shuffled_X_train_car = X_train_car[permutation]
    shuffled_Y_train_car = Y_train_car[permutation]
    
    permutation_test = list(np.random.permutation(size_car_samples_test))
    shuffled_X_test_car = X_test_car[permutation]
    shuffled_Y_test_car = Y_test_car[permutation]


    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_subsamples = math.floor(
        size_car_samples / avg_num_smpl_minor_classes)
 # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_subsamples):
        mini_subsample_X = X_train_car[k * avg_num_smpl_minor_classes: k * avg_num_smpl_minor_classes + avg_num_smpl_minor_classes]
        mini_subsample_Y = Y_train_car[k * avg_num_smpl_minor_classes: k * avg_num_smpl_minor_classes + avg_num_smpl_minor_classes]


        X_sub_sample = np.append(X_train_others, mini_subsample_X, axis=0)
        Y_sub_sample = np.append(Y_train_others, mini_subsample_Y, axis=0)
        X_sub_sample = np.rec.array(X_sub_sample)
        Y_sub_sample = np.rec.array(Y_sub_sample)


        sub_sample = (X_sub_sample, Y_sub_sample)
        sub_samples_train.append(sub_sample)

    if size_car_samples % avg_num_smpl_minor_classes != 0:

        mini_subsample_X = shuffled_X_train_car[num_complete_subsamples * avg_num_smpl_minor_classes: size_car_samples]
        mini_subsample_Y = shuffled_Y_train_car[num_complete_subsamples * avg_num_smpl_minor_classes: size_car_samples]

        X_sub_sample = np.append(X_train_others, mini_subsample_X, axis=0)
        Y_sub_sample = np.append(Y_train_others, mini_subsample_Y,axis=0)
        X_sub_sample = np.rec.array(X_sub_sample)
        Y_sub_sample = np.rec.array(Y_sub_sample)

        sub_sample = (X_sub_sample, Y_sub_sample)
        sub_samples_train.append(sub_sample)

    num_complete_subsamples_test = math.floor(
        size_car_samples_test / avg_num_smpl_minor_classes_test)
    for k in range(0, num_complete_subsamples_test):
        mini_subsample_X = shuffled_X_test_car[k * avg_num_smpl_minor_classes_test: k * avg_num_smpl_minor_classes_test + avg_num_smpl_minor_classes_test]
        mini_subsample_Y = shuffled_Y_test_car[k * avg_num_smpl_minor_classes_test: k * avg_num_smpl_minor_classes_test + avg_num_smpl_minor_classes_test]

        X_sub_sample = np.append(X_test_others, mini_subsample_X, axis=0)
        Y_sub_sample = np.append(Y_test_others, mini_subsample_Y, axis=0)
        X_sub_sample = np.rec.array(X_sub_sample)
        Y_sub_sample = np.rec.array(Y_sub_sample)

        sub_sample = (X_sub_sample, Y_sub_sample)
        sub_samples_test.append(sub_sample)

    if size_car_samples % avg_num_smpl_minor_classes_test != 0:

        mini_subsample_X = shuffled_X_test_car[num_complete_subsamples_test * avg_num_smpl_minor_classes_test: size_car_samples]
        mini_subsample_Y = shuffled_Y_test_car[num_complete_subsamples_test * avg_num_smpl_minor_classes_test: size_car_samples]

        X_sub_sample = np.append(X_test_others, mini_subsample_X, axis=0)
        Y_sub_sample = np.append(Y_test_others, mini_subsample_Y,axis=0)
        X_sub_sample = np.rec.array(X_sub_sample)
        Y_sub_sample = np.rec.array(Y_sub_sample)

        sub_sample = (X_sub_sample, Y_sub_sample)
        sub_samples_test.append(sub_sample)



    # np.save("./resampled_data/sub_samples_CNN_labels_data_4channels_restericted2", sub_samples)


    return sub_samples_train, sub_samples_test


def random_mini_batches(X, Y, Y_train_labels, Y_test_labels, mini_batch_size, class_weight_calculation = 0):

    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    Y_train_labels = np.rec.array(Y_train_labels)
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    mini_batches_weights = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation]
    shuffled_Y_labels =  Y_train_labels[permutation]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y_labels = shuffled_Y_labels[k * mini_batch_size : k * mini_batch_size + mini_batch_size]

        if class_weight_calculation == 0:
            # size_mini_batch= min(mini_batch_size, mini_batch_X.shape[0])
            mini_batch_weight = np.ones(mini_batch_size, dtype=int)

        elif class_weight_calculation == 1:
            #####This part calculate the weights based on mode share inside each minibatches#######
            # decode the onehot array of minibatch

            # find the number of occurrence for each class
            classes, counts = np.unique(mini_batch_Y_labels.class_label, return_counts=True)
            class_counter = dict(zip(classes, counts))

            # classes_test, counts_test = np.unique(Y_test_labels.class_label, return_counts=True)
            # class_counter_test = dict(zip(classes_test, counts_test))
            #
            # decoded_one_hot= np.argmax(mini_batch_Y, axis=1)
            # # find the number of occurrence for each class
            # unique, counts = np.unique(decoded_one_hot, return_counts=True)
            # class_counter = dict(zip(unique, counts))
            # create a vector for weights
            mini_batch_weight = np.zeros(mini_batch_size, dtype = float)
            # convert the type to float
            #mini_batch_weight = mini_batch_weight.astype(float)

            batch_size = mini_batch_size

            for label, count in class_counter.items():
                mini_batch_weight[np.where(mini_batch_weight == label)] = batch_size / float(count)

        else:
            #####This part calculate the weights based on mode share in the whole dataset#######
            # decode the onehot array of Whole datset labels
            # decoded_one_hot = np.argmax(Y, axis=1)
            # # find the number of occurrence for each class
            # unique, counts = np.unique(decoded_one_hot, return_counts=True)
            # class_counter = dict(zip(unique, counts))

            classes, counts = np.unique(Y_train_labels.class_label, return_counts=True)
            class_counter = dict(zip(classes, counts))

            # create a vector for weights
            mini_batch_weight = np.zeros(mini_batch_size, dtype=float)
            #here batch_size for calculating the wiehgts is the number of training examples
            batch_size = m

            for label, count in class_counter.items():
                mini_batch_weight[np.where(mini_batch_weight == label)] = batch_size / float(count)


        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
        mini_batch_weight = mini_batch_weight.tolist()
        mini_batches_weights.append(mini_batch_weight)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y_labels = shuffled_Y_labels[num_complete_minibatches * mini_batch_size : m]

        if class_weight_calculation == 0:
            mini_batch_weight = np.ones(mini_batch_Y.shape[0], dtype=int)

        elif class_weight_calculation == 1:
            #####This part calculate the weights based on minibatches#######
            # decode the onehot array of minibatch
            classes, counts = np.unique(Y_train_labels.class_label, return_counts=True)
            class_counter = dict(zip(classes, counts))

            # classes_test, counts_test = np.unique(Y_test_labels.class_label, return_counts=True)
            # class_counter_test = dict(zip(classes_test, counts_test))
            #
            # decoded_one_hot= np.argmax(mini_batch_Y, axis=1)
            # # find the number of occurrence for each class
            # unique, counts = np.unique(decoded_one_hot, return_counts=True)
            # class_counter = dict(zip(unique, counts))
            # create a vector for weights

            # convert the type to float
            # mini_batch_weight = mini_batch_weight.astype(float)
            mini_batch_weight = np.zeros(mini_batch_Y.shape[0], dtype=float)
            batch_size = mini_batch_size

            for label, count in class_counter.items():
                mini_batch_weight[np.where(mini_batch_weight == label)] = batch_size / float(count)

        else:
            #####This part calculate the weights based on whole dataset#######
            # decode the onehot array of Whole datset labels
            classes, counts = np.unique(Y_train_labels.class_label, return_counts=True)
            class_counter = dict(zip(classes, counts))
            # create a vector for weights
            mini_batch_weight = np.zeros(mini_batch_Y.shape[0], dtype=float)
            #here batch_size for calculating the wiehgts is the number of training examples
            batch_size = m

            for label, count in class_counter.items():
                mini_batch_weight[np.where(mini_batch_weight == label)] = batch_size / float(count)


        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
        mini_batch_weight = mini_batch_weight.tolist()
        mini_batches_weights.append(mini_batch_weight)
    
    return mini_batches, mini_batches_weights


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def forward_propagation_for_predict(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3'] 
                                                           # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3
    
    return Z3

def predict(X, parameters):
    
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    
    x = tf.placeholder("float", [12288, 1])
    
    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)
    
    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
        
    return prediction

#def predict(X, parameters):
#    
#    W1 = tf.convert_to_tensor(parameters["W1"])
#    b1 = tf.convert_to_tensor(parameters["b1"])
#    W2 = tf.convert_to_tensor(parameters["W2"])
#    b2 = tf.convert_to_tensor(parameters["b2"])
##    W3 = tf.convert_to_tensor(parameters["W3"])
##    b3 = tf.convert_to_tensor(parameters["b3"])
#    
##    params = {"W1": W1,
##              "b1": b1,
##              "W2": W2,
##              "b2": b2,
##              "W3": W3,
##              "b3": b3}
#
#    params = {"W1": W1,
#              "b1": b1,
#              "W2": W2,
#              "b2": b2}    
#    
#    x = tf.placeholder("float", [12288, 1])
#    
#    z3 = forward_propagation(x, params)
#    p = tf.argmax(z3)
#    
#    with tf.Session() as sess:
#        prediction = sess.run(p, feed_dict = {x: X})
#        
#    return prediction