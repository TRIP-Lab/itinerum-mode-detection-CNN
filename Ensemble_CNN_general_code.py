#!/usr/bin/env python
"""
Mode Detection with Convolutional Neural Network
Implemented in Tensorflow Library (Version 1.6 installed with Anaconda on Windows 10)
The code read the data files from PostgreSQL database
Please find the 'points.csv' and 'labels.csv' on Github and import them into a PostgreSQL db, or
modify the code to read all the data from csv files directly.
"""
# ==============================================================================
__author__ = "Ali Yazdizadeh"
__date__ = "February 2018"
__email__ = "ali.yazdizadeh@mail.concordia.ca"
__python_version__ = "3.5.4"
# ==============================================================================

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.framework import ops
from sklearn.model_selection import train_test_split
import time
import math
from cnn_utils import random_mini_batches
import pickle


start_time = time.time()
###define the model parameters###

#minibatch size
minibatch_size = 16

#type :
Model_type = 79

#size of trip trajectories
seg_size = 70

# Number of channels
num_channels = 5
# Number of classes
num_classes = 4
#Network architecture atributes
num_channels_ensemble = [5]
num_filters_ensemble = []
filters_size_ensemble = []
num_stride_maxpool_ensemble = []
num_stride_conv2d_ensemble = []
maxpool_size_ensemble = []

num_epoch_values = [100]

num_layers_ensemble = [5]
num_networks = len(num_layers_ensemble)

filters_size_ensemble.append([8,8,8,8,8])


#
num_filters_ensemble.append([96,256,384,384,256])

maxpool_size_ensemble.append([8,8,8,8,8])
for i in range(len(num_layers_ensemble)):
    num_stride_conv2d_ensemble.append([2 for k in range(0, num_layers_ensemble[i])])

for i in range(len(num_layers_ensemble)):
    num_stride_maxpool_ensemble.append([2 for k in range(0, num_layers_ensemble[i])])

weights_ensemble = []
for i in range(len(filters_size_ensemble)):

    filters_size = filters_size_ensemble[i]
    num_filters = num_filters_ensemble[i]

    weights = []
    for index, f in enumerate(filters_size):
        if index == 0:
            weights.append([f, num_channels, num_filters[index]])
        else:
            weights.append([f, num_filters[index - 1], num_filters[index]])

    weights_ensemble.append(weights)


def get_save_path(net_number):
    return save_dir + 'network' + str(net_number)


#######FUNCTIONS
def segmentation(points, seg_size, num_points_per_trip):
    i = 0
    points_segmented = pd.DataFrame()
    # give same size for each segment
    for index, row in num_points_per_trip.iterrows():
        if row[2] < 4:
            continue
        segment_counter = 0

        trip = points.loc[(points['uuid'] == row[0]) & (points['trip_id'] == row[1])]
        num_segs = math.ceil(row[2] / seg_size)

        padding = (-trip.shape[0]) % seg_size
        splitted_trip = np.array_split(np.concatenate((trip, np.zeros((padding, trip.shape[1])))), num_segs)
        for j in range(0, num_segs):
            segment_counter += 1
            trip = pd.DataFrame(data=splitted_trip[j], columns=points.columns.values)
            trip = trip.assign(segment_id=segment_counter)
            points_segmented = points_segmented.append(trip, ignore_index=False)

    # drop trips with na or zero values in 'uuid','trip_id','segment_id'
    points_segmented = points_segmented.dropna(subset=['uuid', 'trip_id', 'segment_id'])
    points_segmented = points_segmented[(points_segmented['uuid'] != 0) &
                                        (points_segmented['trip_id'] != 0) &
                                        (points_segmented['segment_id'] != 0)]

    return points_segmented


#############Preparing the X and Y data to feed to neural net##########
def XY_preparation(points_segmented, labels, seg_size, num_channels):
    # Flatten the training and test sets

    num_segements = \
        points_segmented.drop_duplicates(subset=('uuid', 'trip_id', 'segment_id'), keep='first', inplace=False).shape[0]
    print('num segments', num_segements)

    uuid_trip_id_segments = \
        points_segmented.drop_duplicates(subset=('uuid', 'trip_id', 'segment_id'), keep='first', inplace=False)[
            ['uuid', 'trip_id', 'segment_id']]

    print('uuid_trip_id_segments shape is')
    print(uuid_trip_id_segments.shape)


    # define numpy structured array ('https://docs.scipy.org/doc/numpy-1.10.0/user/basics.rec.html')
    Y_orig_new = np.zeros(num_segements,
                          dtype=[('uuid', 'S64'), ('trip_id', 'int8'), ('segment_id', 'int8'), ('class_label', 'int8')])
    Y_orig_new = np.rec.array(Y_orig_new)
    X_orig_new = np.zeros(num_segements, dtype=[('uuid', 'S64'), ('trip_id', 'int8'), ('segment_id', 'int8'),
                                                ('channels', '(70,5)float64')])
    X_orig_new = np.rec.array(X_orig_new)

    i = 0
    # assign the label for each trip
    #
    for index in range(1, num_channels + 1):
        globals()['Channel_{}'.format(index)] = np.zeros((seg_size))


    for index, row in uuid_trip_id_segments.iterrows():
        # select all the points for each segment
        trip = points_segmented.loc[(points_segmented['uuid'] == row[0]) &
                                    (points_segmented['trip_id'] == row[1]) &
                                    (points_segmented['segment_id'] == row[2])]

        # aasing the labels to each segment
        label = labels.loc[(labels['uid'] == row[0]) & (labels['trip_id'] == row[1])]
        label = np.array(label, dtype=pd.Series)
        if math.isnan(label[0][2]) or label[0][2] > 3:
            continue

        # copy the uuid, trip_id, segment_id and mode of transport to the Y_orig_new
        Y_orig_new[i].class_label = int(label[0][2])
        Y_orig_new[i].uuid = row[0]
        Y_orig_new[i].trip_id = row[1]
        Y_orig_new[i].segment_id = row[2]


        for num, channel in enumerate(channels):

            X_orig_new[i].channels[0:trip.shape[0], num] = trip.loc[:, '{}'.format(channel)]

            X_orig_new[i].uuid = row[0]
            X_orig_new[i].trip_id = row[1]
            X_orig_new[i].segment_id = row[2]



        i += 1
    X_orig_new.channels = np.nan_to_num(X_orig_new.channels, copy=False)

    np.save("./data/augmenteddata_7channels_filtering/CNN_labels_augmenteddata_7channels", Y_orig_new)
    np.save("./data/augmenteddata_7channels_filtering/CNN_segmented_augmenteddata_7channels", X_orig_new)
    print('Data files are saved')

    return (X_orig_new, Y_orig_new)


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


def filtering(points, num_points_per_trip, windows_size=15):
    """
    :param X_orig:
    :return: X_orig filtered high errors with savitzky_golay alg.
    """
    for index, row in num_points_per_trip.iterrows():
        if row[2] < windows_size:
            continue
        temp_speed = np.copy(
            points.loc[(points['uuid'] == row[0]) & (points['trip_id'] == row[1])]['speed_float'].values)
        temp_acceleration = np.copy(
            points.loc[(points['uuid'] == row[0]) & (points['trip_id'] == row[1])]['acceleration'].values)
        temp_jerk = np.copy(points.loc[(points['uuid'] == row[0]) & (points['trip_id'] == row[1])]['jerk'].values)
        temp_bearing_rate = np.copy(
            points.loc[(points['uuid'] == row[0]) & (points['trip_id'] == row[1])]['bearing_rate'].values)

        points.loc[(points['uuid'] == row[0]) & (points['trip_id'] == row[1]), 'speed_float'] = \
            savitzky_golay(temp_speed, window_size=15, order=4)
        points.loc[(points['uuid'] == row[0]) & (points['trip_id'] == row[1]), 'acceleration'] = \
            savitzky_golay(temp_acceleration, window_size=15, order=4)
        points.loc[(points['uuid'] == row[0]) & (points['trip_id'] == row[1]), 'jerk'] = \
            savitzky_golay(temp_jerk, window_size=15, order=4)
        points.loc[(points['uuid'] == row[0]) & (points['trip_id'] == row[1]), 'bearing_rate'] = \
            savitzky_golay(temp_bearing_rate, window_size=15, order=4)

    return points


######################split data to train-test######################
def split_train_test(X_origin, Y_orig):
    X_train_orig, X_test_orig, Y_train_orig, Y_test_orig = train_test_split(X_origin, Y_orig, test_size=0.20,
                                                                            random_state=None)

    return (X_train_orig, X_test_orig, Y_train_orig, Y_test_orig)


######################Convert labes vector to one-hot######################
def convert_to_one_hot(Y, C):
    Y_onehot = np.zeros(Y.shape[0], dtype=[('uuid', 'S64'), ('trip_id', 'int8'), ('segment_id', 'int8'),
                                           ('class_label', '(4,)int8')])
    Y_onehot = np.rec.array(Y_onehot)
    Y_onehot.uuid = Y.uuid
    Y_onehot.trip_id = Y.trip_id
    Y_onehot.segment_id = Y.segment_id
    Y_onehot.class_label = np.eye(C)[Y.class_label.reshape(-1)]

    return Y_onehot


#############Create place holders for X and Y in tensorflow#################
def create_placeholders(seg_size, num_channels, num_classes, minibatch_size):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    seg_size -- trajectory segment size
    num_channels -- number of attributes used for each trajectory segment(distance, speed, acceleration, jerk, bearing rate)
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes

    Returns:
    X -- placeholder for the data input
    Y -- placeholder for the input labels
    """

    ### START CODE HERE ### (approx. 2 lines)
    X = tf.placeholder(tf.float32, shape=(None, seg_size, num_channels))
    Y = tf.placeholder(tf.float32, shape=[None, num_classes])
    minibatch_weights = tf.placeholder(tf.float32, shape=[None])

    return X, Y, minibatch_weights


################initialize the parameters#################
def initialize_parameters(weights):
    """
    Initializes weight parameters to build a neural network with tensorflow.
    Returns:
    parameters -- a dictionary of tensors containing W1, W2, W3 , ...
        """

    # define the parameters for conv layers
    parameters = {}

    for index, current_layer in enumerate(weights):
        # declare 'W's
        globals()['W{}'.format(index + 1)] = tf.get_variable('W{}'.format(index + 1),
                                                             current_layer,
                                                             initializer=tf.contrib.layers.xavier_initializer())

        parameters['W{}'.format(index + 1)] = globals()['W{}'.format(index + 1)]

    return parameters


####################Forward propagation in tensorflow#########################
def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> ........ -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    for index, param in enumerate(parameters):

        # Retrieve the parameters from the dictionary "parameters"
        if index == 0:
            globals()['W{}'.format(index + 1)] = parameters['W{}'.format(index + 1)]

            # CONV2D: stride from num_stride_conv2d, padding 'SAME'
            globals()['Z{}'.format(index + 1)] = tf.nn.conv1d(X, filters=globals()['W{}'.format(index + 1)]
                                                              , stride=num_stride_conv2d[index],
                                                              padding='SAME')

            # RELU
            globals()['A{}'.format(index + 1)] = tf.nn.leaky_relu(globals()['Z{}'.format(index + 1)], alpha=0.02)
            # tf.nn.relu(globals()['Z{}'.format(index + 1)])

            # filter = tf.get_variable('weights', [5, 5, 1, 64],
            #                          initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32),
            #                          dtype=tf.float32)

            # MAXPOOL: window size form stride from num_stride_maxpool, sride is the same size as window size, padding 'SAME'
            globals()['P{}'.format(index + 1)] = tf.layers.max_pooling1d(globals()['A{}'.format(index + 1)],
                                                                         pool_size=maxpool_size[index],
                                                                         strides=num_stride_maxpool[index],
                                                                         padding='SAME')
        else:
            globals()['W{}'.format(index + 1)] = parameters['W{}'.format(index + 1)]

            # CONV2D: stride from num_stride_conv2d, padding 'SAME'
            globals()['Z{}'.format(index + 1)] = tf.nn.conv1d(globals()['P{}'.format(index)],
                                                              filters=globals()['W{}'.format(index + 1)]
                                                              , stride=num_stride_conv2d[index], padding='SAME')

            # RELU
            globals()['A{}'.format(index + 1)] = tf.nn.leaky_relu(globals()['Z{}'.format(index + 1)], alpha=0.02)
            # tf.nn.relu(globals()['Z{}'.format(index + 1)])

            # MAXPOOL: window size form stride from num_stride_maxpool, sride is the same size as window size, padding 'SAME'
            globals()['P{}'.format(index + 1)] = tf.layers.max_pooling1d(globals()['A{}'.format(index + 1)],
                                                                         pool_size=maxpool_size[index],
                                                                         strides=num_stride_maxpool[index],
                                                                         padding='SAME')

    # FLATTEN
    globals()['P{}'.format(len(parameters))] = tf.contrib.layers.flatten(globals()['P{}'.format(len(parameters))])


    # one fully connected layer
    globals()['Z{}'.format(len(parameters) + 1)] = tf.contrib.layers.fully_connected(
        globals()['P{}'.format(len(parameters))], num_classes, activation_fn=None)

    # for index, param in enumerate(parameters):
    #     print(globals()['Z{}'.format(index + 1)])
    #
    # print(globals()['Z{}'.format(len(parameters) + 1)])
    # print(globals()['P{}'.format(len(parameters))])

    final_Z = globals()['Z{}'.format(len(parameters) + 1)]
    return final_Z


####################Computing Cost with softmax_cross_entropy in tensorflow#########################
def compute_cost(final_Z, Y, cl_weights):
    """
    class_weights
    Computes the cost

    Arguments:
    Final_Z -- output of forward propagation
    Y -- "true" labels vector placeholder, same shape as final_Z

    Returns:
    cost - Tensor of the cost function
    """
    cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=final_Z, weights=cl_weights))
    return cost


####################Training the neural net model in Tensorflow#########################
def model(X_train, Y_train, X_test, Y_test, Y_train_labels, Y_test_labels, seg_size, weights,
          decay_learning_rate=False,
          base_learning_rate=0.0001,
          num_epochs=100, minibatch_size=16, print_cost=True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

    Arguments:
    X_train -- training set
    Y_train -- test set
    X_test -- training set
    Y_test -- test set
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables

    (m, seg_size, n_C0) = (X_train.channels.shape)

    # calculating the learning rate type
    batch = tf.Variable(0, dtype=tf.float32)
    if decay_learning_rate is True:

        # Decay once per epoch, using an exponential schedule starting at 0.01.
        learning_rate = tf.train.exponential_decay(
            0.01,  # Base learning rate.
            batch * minibatch_size,  # Current index into the dataset.
            m,  # Decay step.
            0.95,  # Decay rate.
            staircase=True)
    else:
        # Decay once per epoch, using an exponential schedule starting at 0.01.
        learning_rate = tf.train.exponential_decay(
            base_learning_rate,  # Base learning rate.
            batch * minibatch_size,  # Current index into the dataset.
            m,  # Decay step.
            1.00,  # Decay rate.
            staircase=True)

    n_y = num_classes
    costs = []

    # variable for saving trip_id, uuid, segment_id fro test set

    predicted_label = np.zeros(Y_test.shape[0], dtype=[('uuid', 'S64'), ('trip_id', 'int8'), ('segment_id', 'int8'),
                                                       ('class_label', 'int8')])
    predicted_label = np.rec.array(predicted_label)
    predicted_label.uuid = Y_test.uuid
    predicted_label.trip_id = Y_test.trip_id
    predicted_label.segment_id = Y_test.segment_id

    # variable for saving trip_id, uuid, segment_id for train set
    predicted_label_train = np.zeros(Y_train.shape[0], dtype=[('uuid', 'S64'), ('trip_id', 'int8'), ('segment_id',
                                                                                                     'int8'),
                                                              ('class_label', 'int8')])
    predicted_label_train = np.rec.array(predicted_label_train)
    predicted_label_train.uuid = Y_train.uuid
    predicted_label_train.trip_id = Y_train.trip_id
    predicted_label_train.segment_id = Y_train.segment_id


    # Create Placeholders of the correct shape
    X, Y, minibatch_weights = create_placeholders(seg_size, n_C0, n_y, minibatch_size=minibatch_size)

    # Initialize parameters
    parameters = initialize_parameters(weights)
    # Forward propagation: Build the forward propagation in the tensorflow graph
    final_Z = forward_propagation(X, parameters)

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(final_Z, Y, minibatch_weights)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=batch)

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set

            minibatches, mini_batches_weights = random_mini_batches(X_train, Y_train, Y_train_labels, Y_test_labels,
                                                                    minibatch_size, class_weight_calculation=0)

            for index, minibatch in enumerate(minibatches):
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                class_weights = mini_batches_weights[index]


                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                _, temp_cost = sess.run([optimizer, cost],
                                        feed_dict={X: minibatch_X.channels, Y: minibatch_Y.class_label,
                                                   minibatch_weights: class_weights})
                # _, temp_cost = sess.run(tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(compute_cost),
                #                         feed_dict={X: minibatch_X, Y: minibatch_Y, class_weights: class_weights})


                minibatch_cost += temp_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)

            if minibatch_cost > costs[epoch - 1]:
                learning_rate = learning_rate * 0.95

        # Calculate the correct predictions
        predict_op = tf.argmax(final_Z, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # finding the probabilities
        final_prob_train = (final_Z.eval({X: X_train.channels, Y: Y_train.class_label}))


        final_prob_test = sess.run(final_Z,
                                            feed_dict={X: X_test.channels})

        # finding the predicted labels
        predictions, labels_test = sess.run([predict_op, tf.argmax(Y, 1)],
                                            feed_dict={X: X_test.channels, Y: Y_test.class_label})
        # predicted_label.class_label = predictions

        predictions_train = (predict_op.eval({X: X_train.channels, Y: Y_train.class_label}))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        train_accuracy = accuracy.eval({X: X_train.channels, Y: Y_train.class_label})
        test_accuracy = accuracy.eval({X: X_test.channels, Y: Y_test.class_label})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        confusion = tf.confusion_matrix(labels=tf.argmax(Y, 1), predictions=predict_op, num_classes=num_classes)
        confusion_mat = confusion.eval({Y: Y_test.class_label, X: X_test.channels})

        print(confusion_mat)


        return parameters, test_accuracy, costs, predictions, predictions_train, final_prob_train, final_prob_test


if __name__ == '__main__':

    #loading the data
    X_train = np.load(
        "D:/OneDrive - Concordia University - Canada/PycharmProjects/Itinerum-Deep-Neural-Network/data/augmenteddata_5channels/X_train_augmenteddata_5channels.npy")
    X_test = np.load(
        "D:/OneDrive - Concordia University - Canada/PycharmProjects/Itinerum-Deep-Neural-Network/data/augmenteddata_5channels/X_test_cleaned.npy")

    Y_train = np.load(
        "D:/OneDrive - Concordia University - Canada/PycharmProjects/Itinerum-Deep-Neural-Network/data/augmenteddata_5channels/Y_train_augmenteddata_5channels.npy")
    Y_test = np.load(
        "D:/OneDrive - Concordia University - Canada/PycharmProjects/Itinerum-Deep-Neural-Network/data/augmenteddata_5channels/Y_test_cleaned.npy")
    print("X Y train test npy data files are loaded successfully from disc")

    # this part of the code remove the rows from X_train without uuid
    conter_x_test_temp = 0
    for i in range(X_train.shape[0]):
        if len(X_train[i][0]) != 0:
            conter_x_test_temp += 1
    X_train_temp = np.zeros(conter_x_test_temp, dtype=[('uuid', 'S64'), ('trip_id', 'int8'), ('segment_id', 'int8'),
                                                       ('channels', '(70,5)float64')])

    Y_train_temp = np.zeros(conter_x_test_temp, dtype=[('uuid', 'S64'), ('trip_id', 'int8'), ('segment_id',
                                                                                              'int8'),
                                                       ('class_label', 'int8')])

    conter_x_test_temp = 0
    for i in range(X_train.shape[0]):
        if len(X_train[i][0]) == 0:
            continue

        for j in range(3):
            X_train_temp[conter_x_test_temp][j] = np.copy(X_train[i][j])
        X_train_temp[conter_x_test_temp][3] = np.copy(np.delete(X_train[i][3], obj=(5, 6), axis=1))
        Y_train_temp[conter_x_test_temp] = np.copy(Y_train[i])
        conter_x_test_temp += 1
    X_train = np.copy(X_train_temp)
    Y_train = np.copy(Y_train_temp)

    # this part of the code remove the rows from X_test without uuid
    conter_x_test_temp = 0
    for i in range(X_test.shape[0]):
        if len(X_test[i][0]) != 0:
            conter_x_test_temp += 1
    X_test_temp = np.zeros(conter_x_test_temp, dtype=[('uuid', 'S64'), ('trip_id', 'int8'), ('segment_id', 'int8'),
                                                      ('channels', '(70,5)float64')])

    Y_test_temp = np.zeros(conter_x_test_temp, dtype=[('uuid', 'S64'), ('trip_id', 'int8'), ('segment_id',
                                                                                             'int8'),
                                                      ('class_label', 'int8')])
    conter_x_test_temp = 0
    for i in range(X_test.shape[0]):
        if len(X_test[i][0]) == 0:
            continue
        for j in range(3):
            X_test_temp[conter_x_test_temp][j] = np.copy(X_test[i][j])
        X_test_temp[conter_x_test_temp][3] = np.copy(np.delete(X_test[i][3], obj=(5, 6), axis=1))
        Y_test_temp[conter_x_test_temp] = np.copy(Y_test[i])
        conter_x_test_temp += 1

    X_test = np.copy(X_test_temp)
    Y_test = np.copy(Y_test_temp)

    Y_train = np.rec.array(Y_train)
    Y_test = np.rec.array(Y_test)
    X_train = np.rec.array(X_train)
    X_test = np.rec.array(X_test)
    Y_train_labels = np.recarray.copy(Y_train)
    Y_test_labels = np.recarray.copy(Y_test)

    Y_train = convert_to_one_hot(Y_train, num_classes)
    Y_test = convert_to_one_hot(Y_test, num_classes)

    (m, seg_size, n_C0) = (X_train.channels.shape)

    # (m, seg_size, n_C0) = X_train_temp.shape

    n_y = num_classes

    test_accuracies = []
    costs_dict = {}
    test_accuracy_dict = {}

    X, Y, minibatch_weights = create_placeholders(seg_size, n_C0, n_y, minibatch_size=minibatch_size)
    sess = tf.Session()
    save_dir = './checkpoints/'

    ##CREATE A PANDAS DF TO RECORD THE PREDICTED LABELS OF TEST SAMPLES FROM EACH MODEL BELOW
    predicted_label = np.zeros(Y_test_labels.shape[0], dtype=[('uuid', 'S64'), ('trip_id', 'int8'), ('segment_id', 'int8'),
                                                       ('True_label', 'int8')])
    predicted_label = np.rec.array(predicted_label)


    predicted_label_df = pd.DataFrame(data= predicted_label)

    # predicted_label = np.rec.array(predicted_label)
    predicted_label_df['uuid'] = Y_test_labels.uuid
    predicted_label_df['trip_id'] = Y_test_labels.trip_id
    predicted_label_df['segment_id'] = Y_test_labels.segment_id
    predicted_label_df['True_label'] = Y_test_labels.class_label

    predicted_label_train = np.zeros(Y_train.shape[0], dtype=[('uuid', 'S64'), ('trip_id', 'int8'), ('segment_id',
                                                                                                  'int8'),
                                                    ('class_label', 'int8')])
    predicted_label_train = np.rec.array(predicted_label_train)
    predicted_label_train_df = pd.DataFrame(data=predicted_label_train)
    predicted_label_train_df['uuid'] = Y_train_labels.uuid
    predicted_label_train_df['trip_id'] = Y_train_labels.trip_id
    predicted_label_train_df['segment_id'] = Y_train_labels.segment_id
    predicted_label_train_df['True_label'] = Y_train_labels.class_label

    predicted_label_df['uuid'] = predicted_label_df['uuid'].str.decode("utf-8")
    predicted_label_train_df['uuid'] = predicted_label_train_df['uuid'].str.decode("utf-8")

    ##CREATE A PANDAS DF TO RECORD THE PROBABILITIES OF TEST SAMPLES FROM EACH MODEL BELOW
    prob_test = np.zeros(Y_test_labels.shape[0], dtype=[('uuid', 'S64'), ('trip_id', 'int8'), ('segment_id', 'int8'),
                                                       ('True_label', 'int8')])
    predicted_label = np.rec.array(prob_test)

    prob_test_df = pd.DataFrame(data= prob_test)

    # predicted_label = np.rec.array(predicted_label)
    prob_test_df['uuid'] = Y_test_labels.uuid
    prob_test_df['trip_id'] = Y_test_labels.trip_id
    prob_test_df['segment_id'] = Y_test_labels.segment_id
    prob_test_df['True_label'] = Y_test_labels.class_label

    prob_train = np.zeros(Y_train.shape[0], dtype=[('uuid', 'S64'), ('trip_id', 'int8'), ('segment_id','int8'),('class_label', 'int8')])
    prob_train = np.rec.array(prob_train)
    prob_train_df = pd.DataFrame(data=prob_train)
    prob_train_df['uuid'] = Y_train_labels.uuid
    prob_train_df['trip_id'] = Y_train_labels.trip_id
    prob_train_df['segment_id'] = Y_train_labels.segment_id
    prob_train_df['True_label'] = Y_train_labels.class_label

    prob_train_df['uuid'] = prob_train_df['uuid'].str.decode("utf-8")
    prob_test_df['uuid'] = prob_test_df['uuid'].str.decode("utf-8")


    df_length = len(predicted_label['uuid'])

    for epo in num_epoch_values:
        model_architecture_type = 0
        for i in range(num_networks):
            model_architecture_type += 1
            print('CNN Model of type {}'.format(Model_type))
            print()
            print("---------------------------------------------------------")
            print("Neural network type {} Now Generated with epoch {}".format(model_architecture_type, epo))


            # Create a random training-set. Ignore the validation-set.
            filters_size = filters_size_ensemble[i]
            num_filters = num_filters_ensemble[i]
            num_stride_conv2d = num_stride_conv2d_ensemble[i]
            num_stride_maxpool = num_stride_maxpool_ensemble[i]
            maxpool_size = maxpool_size_ensemble[i]
            weights = weights_ensemble[i]

            parameters, test_accuracy, costs, predictions, predictions_train, final_prob_train, final_prob_test = \
                model(X_train, Y_train, X_test, Y_test, Y_train_labels, Y_test_labels, seg_size, weights=weights,
                      decay_learning_rate=False,base_learning_rate=0.0001,num_epochs=epo)

            predicted_label_df['network_{}_epoch{}'.format(model_architecture_type, epo)] = predictions


            predicted_label_train_df['network_{}_epoch{}'.format(model_architecture_type, epo)] = predictions_train


            for col in range(4):
                prob_train_df['network_{}_epoch{}_pob_class_{}'.format(model_architecture_type,
                                                                       epo, col+1)] = pd.Series(final_prob_train[:,col])
            for col in range(4):
                prob_test_df['network_{}_epoch{}_pob_class_{}'.format(model_architecture_type,
                                                                       epo, col+1)] = pd.Series(final_prob_test[:,col])

            costs_dict['network_{}_epoch{}'.format(model_architecture_type, epo)] = costs
            test_accuracy_dict['network_{}_epoch{}'.format(model_architecture_type, epo)] = test_accuracy

            #
            predicted_label_df.to_csv('./results_augmented_data/test_predicted_label_5channels_type100models.csv')
            predicted_label_train_df.to_csv('./results_augmented_data/train_predicted_label_5channels_type100models.csv')
            #
            #
            prob_test_df.to_csv('./results_augmented_data/test_prob_5channels_type100models.csv')
            prob_train_df.to_csv('./results_augmented_data/train_prob_5channels_type100models.csv')


    with open('./results_ensemble/test_accuracy_dict_type17models.pickle', 'wb') as handle:
        pickle.dump(test_accuracy_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./results_ensemble/costs_dict_type17models.pickle', 'wb') as handle:
        pickle.dump(costs_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    print('PREDICTED LABELS AND TEST ACCURACIES ARE SUCCESSFULLY SAVED')
    print("--- %s seconds ---" % (time.time() - start_time))




