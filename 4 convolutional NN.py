# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 10:11:32 2019

@author: Ning
"""
#%% reshape array for convolutional neural network
alldata_CNN = alldata.reshape(m, num_px, num_px, 3)
train_x_CNN, test_x_CNN = train_test_split(
   alldata_CNN, test_size=0.15, random_state=726)
#%% creating placeholders for input
def CNN_create_placeholders(n_H0, n_W0, n_C):
    """
    Creates placeholders for the tensorflow session
    
    Input:
        n_H0: height of the input image
        n_W0: width of the input image
        n_C: number of channels 
    
    Output:
        x: placeholder for data input, of shape [None, n_H0, n_W0, n_C]
        y: placehoder for input label, of shape [None, 1]
    """
    
    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C])
    Y = tf.placeholder(tf.float32, [None, 1])
    
    return(X, Y)
    
# sanity check
tf.reset_default_graph()
x, y = CNN_create_placeholders(64, 64, 3)
print('x:', x)
print('y:', y)

#%% initialize parameters
def CNN_initialize_parameters(seed):
    """
    Initialize parameters for a NN with 2 Convolutional layers for now
    and 3 fully connected layers
    layer 1 filter: [4, 4, 3, 8]
    layer 2 filter: [4, 4, 8, 16]
    # layer 3 filter: [4, 4, 16, 32]
    """
    tf.set_random_seed(seed)
    
    W1 = tf.get_variable('W1', [4,4,3,8], 
                         initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable('W2', [4,4,8,16], 
                         initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    # W3 = tf.get_variable('W3', [4,4,16,32], 
    #                      initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    parameters = {'W1': W1, 'W2': W2#, 'W3':W3
                  }
    return(parameters)
    
# sanity check
tf.reset_default_graph()
parameters = CNN_initialize_parameters(3)
print(parameters)

#%% forward propagation
def CNN_forward_propagation(X, parameters, dropout = False, keep_prob = 0.8):
    """
    Forward propagation step for a NN:
        2 * (conv - relu - max pool) - FC - FC - sigmod
    """
    # unpack the parameters
    W1 = parameters['W1']
    W2 = parameters['W2']
    #W3 = parameters['W3']
    
    # convolutional layers
    Z1 = tf.nn.conv2d(X, W1, strides = [1,1,1,1], padding = 'SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize = [1,8,8,1], strides = [1,8,8,1], padding = 'SAME')
    
    Z2 = tf.nn.conv2d(P1, W2, strides = [1,1,1,1], padding = 'SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')
    
    # flatten
    # Add dropout layer before the fully connected layers to avoid overfitting.
    if dropout == True:
        P2 = dropout_flatten_layer(previous_layer=P2, keep_prob=keep_prob)
    else:
        P2 = tf.contrib.layers.flatten(P2)
    
    # fully connected layers
    A3 = tf.contrib.layers.fully_connected(P2, 8)
    Z4 = tf.contrib.layers.fully_connected(A3, 1, activation_fn = None)
    
    return(Z4)

# sanity check
tf.reset_default_graph()

with tf.Session() as sess:
    np.random.seed(1)
    X, Y = CNN_create_placeholders(64, 64, 3)
    parameters = CNN_initialize_parameters(seed = 3)
    Z4 = CNN_forward_propagation(X, parameters)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(Z4, {X: np.random.randn(2,64,64,3), Y: np.random.randn(2,1)})
    print("Z4 = " + str(a))

#%% Compute cost
def CNN_compute_cost(Z4, Y):
    A4 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits = Z4, labels = Y))
    return(A4)
    
# sanity check
tf.reset_default_graph()

with tf.Session() as sess:
    np.random.seed(1)
    X, Y = CNN_create_placeholders(64, 64, 3)
    parameters = CNN_initialize_parameters(seed=3)
    Z4 = CNN_forward_propagation(X, parameters)
    cost = CNN_compute_cost(Z4, Y)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(cost, {X: np.random.randn(4,64,64,3), Y: np.random.randn(4,1)})
    print("cost = " + str(a))
    
#%% put it together, CNN using minibatch gradient descent
def CNN(train_x, train_y, test_x, test_y, seed, dropout = False, keep_prob = 0.8,
        learning_rate = 0.01, num_epochs = 100, minibatch_size = 64,
        print_cost = True):
    ops.reset_default_graph() 
    (m, n_H0, n_W0, n_C0) = train_x.shape
    #n_y = train_y.shape[0]
    X, Y = CNN_create_placeholders(n_H0, n_W0, n_C0)
    parameters = CNN_initialize_parameters(seed = 3)
    Z4 = CNN_forward_propagation(X, parameters, dropout, keep_prob)
    cost = CNN_compute_cost(Z4, Y)
    # backward propagation
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    # calculate number of minibatches
    num_minibatches = m//minibatch_size + 1
    
    costs = []
    
    # run the tensorflow session
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        
        for epoch in range(num_epochs):
            epoch_cost = 0
            # split into random minibatches
            seed = seed + 1
            minibatches = random_minibatches(
                    train_x, train_y, minibatch_size, seed)
            for minibatch in minibatches:
                (mini_x, mini_y) = minibatch
                _, minibatch_cost = sess.run([optimizer, cost],
                                             feed_dict = {X:mini_x, Y:mini_y})
                epoch_cost += minibatch_cost/num_minibatches
            
            # record cost
            costs.append(epoch_cost)
            # print cost
            if print_cost == True and epoch % 5 == 0:
                print('Cost after {0} epochs is {1}'.format(epoch, epoch_cost))
            
        # plot the costs
        plt.plot(np.squeeze(costs))
        plt.ylabel('Cost')
        plt.xlabel('Epoch')
        plt.title('Learning rate = '+ str(learning_rate))
        plt.show()
        
        # save parameters in a varaible
        parameters = sess.run(parameters)
        
        # calculate the correct predictions
        correct_prediction = tf.equal(tf.greater(tf.sigmoid(Z4),0.5), tf.equal(Y,1))
        
        # calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
        # accuracy on test and training
        print("Training accuracy: ", accuracy.eval({X: train_x, Y: train_y}))
        print("Testing accuracy: ", accuracy.eval({X: test_x, Y: test_y}))
        
    return parameters

#%%
model4 = CNN(train_x_CNN,  train_y, test_x_CNN, test_y, seed = 5)
# Training accuracy:  0.9957356
#  Testing accuracy:  0.6519115            
        
# high accuracy, but high variance.
# improve: 1. not enough data - data augmentation. mirroring, random cropping, color shifting
#          2. add regularization to fully connected layers
#          3. early stopping
        
#%% drop out
# https://github.com/ahmedfgad/CIFAR10CNNFlask/blob/master/Training_CIFAR10_CNN/CIFARTrainCNN.py
def dropout_flatten_layer(previous_layer, keep_prob):
    """
    Applying the dropout layer.
    :param previous_layer: Result of the previous layer to the dropout layer.
    :param keep_prop: Probability of keeping neurons.
    :return: flattened array.
    """
    dropout = tf.nn.dropout(x=previous_layer, keep_prob=keep_prob)
    num_features = dropout.get_shape()[1:].num_elements()
    layer = tf.reshape(previous_layer, shape=(-1, num_features))#Flattening the results.
    return layer

#%% with drop out
model4 = CNN(train_x_CNN,  train_y, test_x_CNN, test_y, seed = 5, dropout = True)