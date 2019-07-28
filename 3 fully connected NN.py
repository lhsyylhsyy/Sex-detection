# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 10:10:09 2019

@author: Ning
"""

#%% tensorflow
# 1. create placeholders for input
def create_placeholders(n_x):
    """
    n_x: num_px * num_px * 3
    """
    X = tf.placeholder(tf.float32, [n_x, None])
    Y = tf.placeholder(tf.float32, [1, None])
    return(X,Y)
# sanity check    
X, Y = create_placeholders(n_x = alldata.shape[0])
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
    m = X.shape[0]
    minibatches = []
    num_minibatches = m//minibatch_size + 1
    # shuffle
    permutation = np.random.permutation(m)
    # for flattened data array, used in CF NN
    if X.ndim == 2:  
        X_perm = X[permutation,:]
        Y_perm = Y[permutation,:]
        # divide into minibatcheds
        # all but the last minibatches
        for b in range(num_minibatches-1):
            X_min = X_perm[minibatch_size*b:minibatch_size*(b+1),:]
            Y_min = Y_perm[minibatch_size*b:minibatch_size*(b+1),:]
            minibatch = (X_min, Y_min)
            minibatches.append(minibatch)
        # the last minibatch
        X_min = X_perm[minibatch_size*(num_minibatches-1):m,:]
        Y_min = Y_perm[minibatch_size*(num_minibatches-1):m,:]
        minibatch = (X_min, Y_min)
        minibatches.append(minibatch)
    # for unflattened data array, used in CNN
    elif X.ndim == 4:
        X_perm = X[permutation,:,:,:]
        Y_perm = Y[permutation,:]
        # divide into minibatcheds
        # all but the last minibatches
        for b in range(num_minibatches-1):
            X_min = X_perm[minibatch_size*b:minibatch_size*(b+1),:,:,:]
            Y_min = Y_perm[minibatch_size*b:minibatch_size*(b+1),:]
            minibatch = (X_min, Y_min)
            minibatches.append(minibatch)
        # the last minibatch
        X_min = X_perm[minibatch_size*(num_minibatches-1):m,:,:,:]
        Y_min = Y_perm[minibatch_size*(num_minibatches-1):m,:]
        minibatch = (X_min, Y_min)
        minibatches.append(minibatch)
    else:
        print('I can\'t do this yet!')
    
    return(minibatches)

# test
minibatches = random_minibatches(train_x,  train_y)
minibatches_CNN = random_minibatches(train_x_CNN,  train_y)
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

# %% run!
parameters = model(train_x,  train_y, test_x, test_y, num_epochs = 2000, seed = 5)
# 500 epochs
# Training accuracy:  0.7462637
# Testing accuracy:  0.68333334
# 2000 epocks, overfitted
# Training accuracy:  0.9837263
# Testing accuracy:  0.6333333
        
parameters_2 = model(train_x,  train_y, test_x, test_y, num_epochs = 2000,
                     hidden_layer_sizes = [100, 25, 5, 1], seed = 3)
# 500 epochs
# Training accuracy:  0.7588841
# Testing accuracy:  0.65
# 2000 epochs, overfitted
# Training accuracy:  0.99933577
# Testing accuracy:  0.63666666