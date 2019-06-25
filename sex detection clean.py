# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:49:25 2019

@author: nx001
"""
#%%
# to install packages directly in Python, not this EDI
# pip install tensorflow
# pip install opencv-python

#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.python.framework import ops
from astropy.visualization import make_lupton_rgb # for making own rgb array
#%% set current working directory
os.chdir("C:\\Users\\Ning\\Google Drive\\python\\sex data")
#%%
# read image files, rescale to 64*64 pixels, normalize 

num_px = 128
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            img_scaled = np.array(Image.fromarray(img).resize((num_px,num_px)))
            img_scaled = img_scaled.reshape((num_px*num_px*3,1))
            img_scaled = img_scaled/255
            images.append(img_scaled)
    return images

men = load_images_from_folder("men")
women = load_images_from_folder('women')

#%% plot the images

i = 6
ima = men[i]*255
image_r = ima.reshape((num_px, num_px, 3))
image = make_lupton_rgb(image_r[:,:,2], image_r[:,:,1], image_r[:,:,0], stretch=0.5)
plt.imshow(image)

#%%
# create array
menarray = np.array(men)
m1 = menarray.shape[0]
n_x = menarray.shape[1]
menarray = menarray.reshape((m1,n_x)).T
print("m1 = " + str(m1))

womenarray = np.array(women)
m2 = womenarray.shape[0]
print("men array and women array dimension match?",n_x == womenarray.shape[1])
womenarray = womenarray.reshape((m2,n_x)).T
print("m2 = " + str(m2))

alldata = np.concatenate([menarray, womenarray], axis = 1)
m = alldata.shape[1]

#%%
# create labels: 1 as men, 0 as women
labels1 = np.ones(len(men))
labels0 = np.zeros(len(women))
all_labels = np.concatenate([labels1, labels0])
len(all_labels)
all_labels = all_labels.reshape([1, m])

#%%   
train_x, test_x, train_y, test_y = train_test_split(
 alldata.T, all_labels.T, test_size=300, random_state=425)
train_x = train_x.T
test_x = test_x.T
train_y = train_y.T
test_y = test_y.T
print("Number of testing examples: " + str(test_x.shape[1]))
print("test_x shape:" + str(test_x.shape))
print("test_y shape:" + str(test_y.shape))
print("train_x shape:" + str(train_x.shape))
print("train_y shape:" + str(train_y.shape))

#%%
# 1. create placeholders for input
def create_placeholders(n_x):
    """
    n_x: num_px * num_px * 3
    """
    X = tf.placeholder(tf.float32, [n_x, None])
    Y = tf.placeholder(tf.float32, [1, None])
    return(X,Y)
# sanity check    
X, Y = create_placeholders(n_x)
print(X)
print(Y)
#%%
# 2. initialize parameters
def initialize_parameters(n_x, hidden_layer_sizes, seed = 1):
    """
    Input:
        n_x: dimension of input variable, num_px * num_px * 3
        minibatch_size
        layer_sizes: a list of length L, the number of hidden layers, 
           that gives the number of units in each layer. 
        seed: the random seed
    Output:
        parameters: A dictionary of initial parameters        
    """
    layer_sizes = [n_x]
    layer_sizes.extend(hidden_layer_sizes)
    L = len(hidden_layer_sizes)
    tf.set_random_seed(seed)
    parameters = {}
    for l in range(L):
        parameters["W" + str(l+1)] = tf.get_variable(name = "W" + str(l+1), 
                  shape = [layer_sizes[l+1], layer_sizes[l]], 
                  initializer = tf.contrib.layers.xavier_initializer(seed = seed+l))
        parameters["b" + str(l+1)] = tf.get_variable(name = "b" + str(l+1),
                  shape = [layer_sizes[l+1], 1], initializer = tf.zeros_initializer())
    return(parameters)
    
# sanity check
parameters = initialize_parameters(n_x , hidden_layer_sizes= [25, 10, 1])

#%%
# 3. forward propagation
def forward_propagation(X, parameters):
    L = len(parameters)//2
    Z = tf.add(tf.matmul(parameters['W1'], X), parameters['b1'])
    for l in range(1, L):
        A = tf.nn.relu(Z)
        Z = tf.add(tf.matmul(parameters['W'+str(l+1)], A), parameters['b'+str(l+1)])
    return(Z)

# sanity check
forward_propagation(X, parameters)

#%%
# 4. compute cost
def compute_cost(Z, Y):
    logits = tf.sigmoid(Z)
    labels = tf.cast(Y, "float")
    cost = - tf.reduce_mean(labels*tf.log(logits) + (1-labels)*tf.log(1-logits))
    return(cost)

# test
z = tf.constant([-1, -.5, .5, 1, 2])
z2 = tf.constant([-2, -1, -.5, 1, 2])
y = tf.constant([0, 0, 0, 1, 1])
with tf.Session() as sess:
    print(sess.run(compute_cost(Z = z, Y= y)))
    print(sess.run(compute_cost(Z = z2, Y= y)))
#%%
# write my own minibatch function
def random_minibatches(X, Y, minibatch_size = 64, seed = 1):
    np.random.seed(seed)
    m = X.shape[1]
    # shuffle
    permutation = np.random.permutation(m)
    X_perm = X[:,permutation]
    Y_perm = Y[:,permutation]
    # divide into minibatcheds
    minibatches = []
    num_minibatches = m//minibatch_size + 1
    # all but the last minibatches
    for b in range(num_minibatches-1):
        X_min = X_perm[:, minibatch_size*b:minibatch_size*(b+1)]
        Y_min = Y_perm[:, minibatch_size*b:minibatch_size*(b+1)]
        minibatch = (X_min, Y_min)
        minibatches.append(minibatch)
    # the last minibatch
    X_min = X_perm[:, minibatch_size*(num_minibatches-1):m]
    Y_min = Y_perm[:, minibatch_size*(num_minibatches-1):m]
    minibatch = (X_min, Y_min)
    minibatches.append(minibatch)
    
    return(minibatches)

# test
minibatches = random_minibatches(train_x,  train_y)
#%%
# 5. build the model
def model(train_x, train_y, test_x, test_y, learning_rate = 0.0001, 
          hidden_layer_sizes = [25, 10, 1],
          num_epochs = 500, minibatch_size = 64, print_cost = True, seed = 1):
    ops.reset_default_graph()
    tf.set_random_seed(seed)
    n_x, m = train_x.shape
    num_minibatches = m//minibatch_size + 1
    costs = []
    
    X, Y = create_placeholders(n_x)
    parameters = initialize_parameters(n_x, hidden_layer_sizes)
    Z = forward_propagation(X, parameters)
    cost = compute_cost(Z, Y)
    optimizer = tf.train.GradientDescentOptimizer(
            learning_rate = learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0
            seed = seed + 1
            minibatches = random_minibatches(train_x, train_y, 
                                             minibatch_size, seed)
            
            for minibatch in minibatches:
                (X_min, Y_min) = minibatch
                _, minibatch_cost = sess.run([optimizer, cost], 
                                             feed_dict = {X:X_min, Y:Y_min})
                epoch_cost += minibatch_cost/num_minibatches
            if print_cost == True and epoch % 100== 0:
                print("Cost after "+ str(epoch) + " epochs: " + str(epoch_cost))
            if print_cost == True and epoch % 10 ==0:
                costs.append(epoch_cost)
                
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        
        # save parameters in a varaible
        parameters = sess.run(parameters)
        
        # calculate the correct predictions
        correct_prediction = tf.equal(tf.greater(tf.sigmoid(Z),0.5), tf.equal(Y,1))
        
        # calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
        # accuracy on test and training
        print("Training accuracy: ", accuracy.eval({X: train_x, Y: train_y}))
        print("Testing accuracy: ", accuracy.eval({X: test_x, Y: test_y}))
        
    return parameters

#%% run!
parameters = model(train_x,  train_y, test_x, test_y, num_epochs = 500, seed = 3)
        

