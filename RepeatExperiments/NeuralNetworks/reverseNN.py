#Inverse neural network as used by Bench-Capon (1993)
#Author: Cor Steging
####

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import math
import sys
import random

import BasicNN as NN

def exportOutputs(outputs, filename):
	keys = outputs[0].keys()
	with open(filename, 'w', newline='') as output:
		dict_writer = csv.DictWriter(output, keys, delimiter=',')
		dict_writer.writeheader()
		dict_writer.writerows(outputs)


#Prints information about the model
def printModel(model):
	print('Hidden weights 1:', model['W_hidden_1'].shape)
	print(model['W_hidden_1'])
	print('Hidden weights 2:', model['W_hidden_2'].shape)
	print(model['W_hidden_2'])
	print('Hidden weights 3:', model['W_hidden_3'].shape)
	print(model['W_hidden_3'])
	print('Output weights:', model['W_output'].shape)
	print(model['W_output'].T)
	print(model['features'])

#Creates a neural network model
def createModel(num_input_nodes, num_hidden_layers, num_hidden_nodes, num_output_nodes, learning_rate, features, output_name):

	np.random.seed(0)

	#Create weights for the hidden layer nodes (+1 for the biases)
	W_hidden_1 = 2*np.random.random((num_input_nodes + 1, num_hidden_nodes[0])) - 1
	
	#Create weights for the second hidden layer nodes (+1 for the biases)
	W_hidden_2 = 2*np.random.random((num_hidden_nodes[0] + 1, num_hidden_nodes[1])) - 1
	
	#Create weights for the third hidden layer nodes (+1 for the biases)
	W_hidden_3 = 2*np.random.random((num_hidden_nodes[1] + 1, num_hidden_nodes[2])) - 1

	#Weights for hidden layer (+1 for the biases)
	W_output = 2*np.random.random((num_hidden_nodes[2] + 1, num_output_nodes)) - 1

	features.remove(output_name)

	#Return the model as a dictionary of the weights
	return {"num_hidden": num_hidden_layers, "W_hidden_1":W_hidden_1,"W_hidden_2":W_hidden_2,"W_hidden_3":W_hidden_3, "W_output":W_output, "learning_rate":learning_rate, "features":features}

def backprop(model, X, target, act_type):
	#Forward propagate
	input_layer_outputs = np.hstack((np.ones((X.shape[0], 1)), X))
	hidden_layer1_outputs = np.hstack((np.ones((X.shape[0], 1)), NN.activation_func(np.dot(input_layer_outputs, model['W_hidden_1']), type=act_type)))
	hidden_layer2_outputs = np.hstack((np.ones((X.shape[0], 1)), NN.activation_func(np.dot(hidden_layer1_outputs, model['W_hidden_2']), type=act_type)))
	hidden_layer3_outputs = np.hstack((np.ones((X.shape[0], 1)), NN.activation_func(np.dot(hidden_layer2_outputs, model['W_hidden_3']), type=act_type)))
	output_layer_outputs = NN.activation_func(np.dot(hidden_layer3_outputs, model['W_output']), type=act_type)

	#Compute the errors
	output_error = output_layer_outputs - target
	local_error = sum(np.absolute(output_error))
	#print(local_error)
	hidden_error3 = NN.activation_func(hidden_layer3_outputs[:, 1:], deriv=True, type=act_type) * np.dot(output_error, model['W_output'].T[:, 1:])
	hidden_error2 = NN.activation_func(hidden_layer2_outputs[:, 1:], deriv=True, type=act_type) * np.dot(hidden_error3, model['W_hidden_3'].T[:, 1:])
	hidden_error1 = NN.activation_func(hidden_layer1_outputs[:, 1:], deriv=True, type=act_type) * np.dot(hidden_error2, model['W_hidden_2'].T[:, 1:])

	# partial derivatives
	hidden_pd1 = input_layer_outputs[:, :, np.newaxis] * hidden_error1[: , np.newaxis, :]
	hidden_pd2 = hidden_layer1_outputs[:, :, np.newaxis] * hidden_error2[: , np.newaxis, :]
	hidden_pd3 = hidden_layer2_outputs[:, :, np.newaxis] * hidden_error3[: , np.newaxis, :]
	output_pd = hidden_layer3_outputs[:, :, np.newaxis] * output_error[:, np.newaxis, :]

	# average for total gradients
	total_hidden1_gradient = np.average(hidden_pd1, axis=0)
	total_hidden2_gradient = np.average(hidden_pd2, axis=0)
	total_hidden3_gradient = np.average(hidden_pd3, axis=0)
	total_output_gradient = np.average(output_pd, axis=0)

	#update weights
	model['W_output'] += total_output_gradient * -model["learning_rate"]
	model['W_hidden_1']  += total_hidden1_gradient * -model["learning_rate"]
	model['W_hidden_2']  += total_hidden2_gradient * -model["learning_rate"]
	model['W_hidden_3']  += total_hidden3_gradient * -model["learning_rate"]
	return local_error
#Trains the model using the default batch system
def trainModelBatch(model, train_set, output_name, act_type, rounds, plot=False):
	#Prepare the dataset by splitting the input from the output
	all_y, all_x = NN.splitIO(train_set, output_name)
	
	#Used to store the error over time
	error = []

	#For each training round
	for i in range(rounds):

		#Print the current round
		sys.stdout.write('\r'+repr(round(100*float(i)/float(rounds), 2)))

		#The error of this round
		local_error = 0

		#Set the target value and the current input
		target = all_y
		X = all_x

		#Backpropagation
		local_error = backprop(model, X, target, act_type)
		
		error.append(local_error/len(all_x))
	NN.plotErrorRate(error, plot, name='3_layer_batch')
	return model

#Trains the NN model using stochastic descent
def trainModelStochastic(model, train_set, output_name, act_type, rounds, plot=False):
	#Prepare the dataset by splitting the input from the output
	all_y, all_x = NN.splitIO(train_set, output_name)
	
	#Used to store the error over time
	error = []

	#For each training round
	for i in range(rounds):

		#Print the current round
		sys.stdout.write('\r'+repr(round(100*float(i)/float(rounds), 2)))
		
		#The error of this round
		local_error = 0

		for idx, cur_x in enumerate(all_x):

			#Set the target value and the current input
			target = all_y[idx]
			X = np.asfarray([cur_x])

			#Backpropagation
			local_error += backprop(model, X, target, act_type)
			
		error.append(local_error/len(all_x))
	NN.plotErrorRate(error, plot, name='3_layer_stochastic')
	return model

#Trains the model using the mini batch approach
def trainModelMiniBatch(model, train_set, batchsize, output_name, act_type, rounds, plot=False):
	#Prepare the dataset by splitting the input from the output
	all_y, all_x = NN.splitIO(train_set, output_name)
	
	#Used to store the error over time
	error = []

	#List of batches
	batches = []

	#The number of batches
	num_batches = math.ceil(len(train_set)/batchsize)

	print(len(train_set), batchsize, num_batches)

	#Crate each batch, consisting of a number of inputs [0] and their output [1]
	for i in range(num_batches):
		batches.append(list([all_x[i*batchsize:(i*batchsize)+batchsize], all_y[i*batchsize:(i*batchsize)+batchsize]]))

	#For each training round
	for r in range(rounds):

		#Print the current round
		sys.stdout.write('\r'+repr(round(100*float(r)/float(rounds), 2)))

		#The error of this round
		local_error = 0

		#For each batch
		for batch in batches:

			#Set the target value and the current input
			X = batch[0]
			target = batch[1]

			#Backpropagation
			local_error += backprop(model, X, target, act_type)
				
		error.append(local_error/len(all_x))
	NN.plotErrorRate(error, plot, name='3_layer_minibatch')
	return model


#Classifies a test set of instances using the NN model
def classify(model, test_set, output_name, show_output, act_type=0):
	print('\nclassifying')
	correct = 0

	classified_instances = []

	#print(test_set)
	for instance in test_set:
		target, X = NN.splitIO([instance], output_name)
		l0 = np.hstack((np.ones((X.shape[0], 1)), X))
		l1 = np.hstack((np.ones((X.shape[0], 1)), NN.activation_func(np.dot(l0, model['W_hidden_1']), type=act_type)))
		l2 = np.hstack((np.ones((X.shape[0], 1)), NN.activation_func(np.dot(l1, model['W_hidden_2']), type=act_type)))
		l3 = np.hstack((np.ones((X.shape[0], 1)), NN.activation_func(np.dot(l2, model['W_hidden_3']), type=act_type)))
		output = NN.activation_func(np.dot(l3, model['W_output']), type=act_type)

		#print('input', l0, 'target:', target, 'output:', round(output[0][0]), output[0])
		if(show_output): print('target:', target, 'output:',output[0])
		instance_copy = instance.copy()
		minusOne = 0
		for idx, key in enumerate(instance):
			print(key)
			if key == output_name:
				minusOne = -1
				continue
			else:
				instance_copy[key] = output[0][idx + minusOne]
		classified_instances.append(instance_copy)
	return classified_instances

if __name__ == "__main__":
	

	#Type of activation function
	act_type = 0

	#Proportion test- to training-set size
	test_prop = 0.2

	#number of hidden layers
	num_hidden_layers = 3

	#number of hidden nodes in the layer
	num_hidden_nodes = [3, 10, 24]

	#The name of the output attribute
	output_name = "satisfied"

	#Read in the database and divide it into a test and train set
	db = NN.readCSV("../datasets/welfare.csv")
	db_onefail =  NN.readCSV("../datasets/welfare_onefail.csv")

	random.shuffle(db)

	test_set = db[0:1]
	train_set = db_onefail

	#Number of input nodes
	num_input_nodes = 1

	#The number of output nodes
	num_output_nodes = len(db[0])-1

	#Learning rate
	learning_rate = 0.05

	model = createModel(num_input_nodes, num_hidden_layers, num_hidden_nodes, num_output_nodes, learning_rate, list(db[0].keys()), output_name)
	printModel(model)
	model = trainModelMiniBatch(model, train_set, 10, output_name, 0, 200, plot=False)
	printModel(model)
	outputs = classify(model, test_set, output_name, True)
	NN.printDBInfo(db)
	print(outputs)
	exportOutputs(outputs, 'Outputs/Reverse/reverseOutputsB.csv')
	outputs = classify(model, test_set, output_name, True)




