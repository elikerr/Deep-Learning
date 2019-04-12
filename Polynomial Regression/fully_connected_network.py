# Fully Connected Networks

def two_layer_network_forward(inputs, parameters, return_intermediary_results = False):
    #This block defines a fully connected two layer network that uses linear regression and ReLu activation
    # parameters were originally initialized by a separate function.
    out_1 =  inputs@parameters['weights_1']+parameters['bias_1']
    out_1_relu =np.maximum(out_1,0)
    out_2 =  out_1_relu@parameters['weights_2']+parameters['bias_2']
    
    #return_intermediary_results should only be True if you are going to use this forward pass
    # to calculate gradients for the network parameters
    if return_intermediary_results:
        to_return = {'out_1': out_1, 'out_1_relu': out_1_relu, 'out_2': out_2}
    else:
        to_return = out_2
    return to_return

##This section of code is intended to work in unison to compute the upstream gradient of the mse loss with respect to 
# each of our parameter elements. It will be used in training and updating the paramter values. 
def mse_loss_backward(predicted, gt):
    
    derivative_of_mse_loss_with_respect_to_predicted = (2./len(predicted))*(predicted-gt)
    
    return derivative_of_mse_loss_with_respect_to_predicted
 
def two_layer_network_backward(inputs, parameters, gt, loss_backward = mse_loss_backward):
    
    intermediary_results_in_forward = two_layer_network_forward(inputs, parameters, return_intermediary_results = True)

    out_1 = intermediary_results_in_forward['out_1'] 
    out_1_relu = intermediary_results_in_forward['out_1_relu'] 
    out_2 = intermediary_results_in_forward['out_2'] 
    
    derivative_of_loss_with_respect_to_out_2 =  loss_backward(out_2, gt) 
   
    derivative_of_loss_with_respect_to_bias_2 =  np.ones((1,len(out_1_relu)))@derivative_of_loss_with_respect_to_out_2
    derivative_of_loss_with_respect_to_weights_2 =  out_1_relu.T@derivative_of_loss_with_respect_to_out_2
    derivative_of_loss_with_respect_to_out_1_relu =  derivative_of_loss_with_respect_to_out_2@(parameters['weights_2'].T)
    derivative_of_loss_with_respect_to_out_1=derivative_of_loss_with_respect_to_out_1_relu*((out_1 >0).astype(int))
    derivative_of_loss_with_respect_to_bias_1 =  np.ones((1,len(inputs)))@derivative_of_loss_with_respect_to_out_1
    derivative_of_loss_with_respect_to_weights_1 = inputs.T@derivative_of_loss_with_respect_to_out_1 
    
    return {
            'weights_1': derivative_of_loss_with_respect_to_weights_1,
            'bias_1': derivative_of_loss_with_respect_to_bias_1,
            'weights_2':derivative_of_loss_with_respect_to_weights_2, 
            'bias_2':derivative_of_loss_with_respect_to_bias_2
            }


def run_batch_sgd(backward_function, parameters, learning_rate, inputs, targets):
	#calculate gradients and update parameters using stochastic gradient descent update rule
	# in coordination with the given upstream backward loss function
    gradient=backward_function(inputs,parameters, targets)
    updated_parameters={
            'weights_1': parameters['weights_1']-learning_rate*gradient['weights_1'],
            'bias_1': parameters['bias_1']-learning_rate*gradient['bias_1'],
            'weights_2':parameters['weights_2']-learning_rate*gradient['weights_2'], 
            'bias_2':parameters['bias_2']-learning_rate*gradient['bias_2']
            }
    return updated_parameters


## This section of code is used to define important hyperparamters for training my fully connected
# model. 
n_hidden_nodes = 50
parameters_two_layer_regression = initialize_parameters_ex2(1, n_hidden_nodes, 1)
initial_parameters=parameters_two_layer_regression
learning_rate = 0.001
batch_size = 16
n_epochs = 1000

#added to track training error throughout training.
errortracker=[]

#This loop will train and update the parameters of our model to better match the training data
for epoch in range(n_epochs):
    shuffled_indexes = (np.arange(x_ex1_train.shape[0]))
    np.random.shuffle(shuffled_indexes)
    shuffled_indexes = shuffled_indexes.reshape([-1, batch_size])
    for batch_i in range(shuffled_indexes.shape[1]):    
        batch = shuffled_indexes[:,batch_i]
        input_this_batch = x_ex1_train[batch,:]
        gt_this_batch =  y_ex1_train[batch,:]
        #run_batch_sgd to update the parameters
        parameters_two_layer_regression=run_batch_sgd(two_layer_network_mse_backward, parameters_two_layer_regression, learning_rate, input_this_batch,gt_this_batch)
    errortracker.append(float(mse(two_layer_network_forward(x_ex1_train, parameters_two_layer_regression),y_ex1_train)))

                            
#plot the results of training
plt.plot(errortracker)
plt.xlabel('Epoch')
plt.ylabel('Training Error')

## Medical Dataset - Classification task with cross-entropy and softmax

def ce_loss(out, target):
    ##Calculates the Cross Entropy loss b/w output vector and target vector that will bu used for this 
    # section of the assignment
    lossmatrix=-target@np.log(out).T
    loss = np.sum(np.diag(lossmatrix))/len(lossmatrix)
    return loss

learning_rate = 0.01 
batch_size = 50
n_epochs = 100

##This block is another training block intended to train another, similar fully connected layer.
# the intent of this exercise was to be able to track and save your best model so that you could plot 
# the progress of training
errortracker=[]
for hidden_nodes in range(1,51):
    parameters_two_layer_regression=initialize_parameters_ex2(train_data_ex3.shape[1],hidden_nodes,train_labels_ex3.shape[1])
    for epoch in range(n_epochs):
        shuffled_indexes = (np.arange(train_data_ex3.shape[0]))
        np.random.shuffle(shuffled_indexes)
        shuffled_indexes = shuffled_indexes.reshape([-1, batch_size])
        for batch_i in range(shuffled_indexes.shape[1]):
            batch = shuffled_indexes[:,batch_i]
            input_this_batch = train_data_ex3[batch,:]
            gt_this_batch =  train_labels_ex3[batch,:]
            #use you function run_batch_sgd to update the parameters
            parameters_two_layer_regression=run_batch_sgd(two_layer_network_softmax_ce_backward, parameters_two_layer_regression, learning_rate, input_this_batch,gt_this_batch)
    model=two_layer_network_softmax_forward(val_data_ex3, parameters_two_layer_regression)
    if hidden_nodes==1:
        bestmodel=parameters_two_layer_regression
        old = count_correct_predictions(model,val_labels_ex3)
        errortracker.append(old)
    else:
        new = count_correct_predictions(model,val_labels_ex3)
        errortracker.append(new)
        if old < new:
            bestmodel=parameters_two_layer_regression
            old=new

        else:
            pass

plt.plot(errortracker)
plt.ylabel('Number of corect predictions')
plt.xlabel('Number of Hidden Nodes')
        
#A test of my best model
modeltest = two_layer_network_softmax_forward(test_data_ex3, bestmodel)
print('The best model that I generated has ' + str(100*count_correct_predictions(modeltest, test_labels_ex3)/len(test_labels_ex3)) + '% error when compared to the test data.')
