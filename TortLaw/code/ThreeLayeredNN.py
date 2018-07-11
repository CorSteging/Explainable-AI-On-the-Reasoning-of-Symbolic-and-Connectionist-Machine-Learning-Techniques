#Basic neural network
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


#Exports the model to files
def exportModel(model, path):
	file = open(path+"model_info.txt", "w")
	num_hidden_layers = model["num_hidden"]
	file.write(repr(num_hidden_layers)+'\n')
	if(num_hidden_layers == 1):
		np.savetxt(path + 'model_1L_W_1.out', model["W_hidden"])
		np.savetxt(path + 'model_1L_W_output.out', model["W_output"])
	if(num_hidden_layers == 2):
		np.savetxt(path + 'model_2L_W_1.out', model["W_hidden_1"])
		np.savetxt(path + 'model_2L_W_2.out', model["W_hidden_2"])
		np.savetxt(path + 'model_2L_W_output.out', model["W_output"])
	if(num_hidden_layers == 3):
		np.savetxt(path + 'model_3L_W_1.out', model["W_hidden_1"])
		np.savetxt(path + 'model_3L_W_2.out', model["W_hidden_2"])
		np.savetxt(path + 'model_3L_W_3.out', model["W_hidden_3"])
		np.savetxt(path + 'model_3L_W_output.out', model["W_output"])
	for f in model['features']:
		file.write(f + ', ')
	file.close() 

#imports the model from files
def importModel(path):
	file = open(path + "model_info.txt", "r")
	model = {}
	data = file.read()
	model["W_hidden_1"] = np.genfromtxt(path + "model_3L_W_1.out", dtype='float')
	model["W_hidden_2"] = np.genfromtxt(path + "model_3L_W_2.out", dtype='float')
	model["W_hidden_3"] = np.genfromtxt(path + "model_3L_W_3.out", dtype='float')
	model["W_output"] = np.genfromtxt(path + "model_3L_W_output.out", dtype='float')
	model['features'] = []
	for x in data:
		if(x in [data[0], ',', ' ', '\n']): 
			continue
		else:
			model['features'].append(x)
	return model

def exportOutputs(outputs, filename):
	for instance in outputs:
		instance['output'] = instance['output'][0]
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
	all_x, all_y = NN.splitIO(train_set, output_name)
	
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
	all_x, all_y = NN.splitIO(train_set, output_name)
	
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
def trainModelMiniBatch1(model, train_set, batchsize, output_name, act_type, rounds, plot=False):
	#Prepare the dataset by splitting the input from the output
	all_x, all_y = NN.splitIO(train_set, output_name)
	
	#Used to store the error over time
	error = []

	#List of batches
	batches = []

	#The number of batches
	num_batches = math.ceil(len(train_set)/batchsize)


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
	sys.stdout.write('\r')
	NN.plotErrorRate(error, plot, name='3_layer_minibatch')
	return model

#Trains the model using the mini batch approach
def trainModelMiniBatch(model, train_set, batchsize, output_name, act_type, rounds, plot=False):
	#Prepare the dataset by splitting the input from the output
	all_x, all_y = NN.splitIO(train_set, output_name)
	
	#Used to store the error over time
	error = []

	#List of batches
	batches = []

	#The number of batches
	num_batches = math.ceil(len(train_set)/batchsize)

	used_rounds = 0

	#Crate each batch, consisting of a number of inputs [0] and their output [1]
	for i in range(num_batches):
		batches.append(list([all_x[i*batchsize:(i*batchsize)+batchsize], all_y[i*batchsize:(i*batchsize)+batchsize]]))

	#For each training round
	for r in range(rounds):

		if r > 10 and error[-1] < 0.01:
			if error[-11] - error[-1] < 0.0001:
				print('\nstopping early: ' + repr(error[-1]))
				used_rounds = r+1
				break


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
		used_rounds = r+1
	sys.stdout.write('\r')
	model['training_instances_used'] = used_rounds * len(train_set)
	return model

#Trains the model using the mini batch approach
def trainModelMiniBatchNoStop(model, train_set, batchsize, output_name, act_type, rounds, plot=False):
	#Prepare the dataset by splitting the input from the output
	all_x, all_y = NN.splitIO(train_set, output_name)
	
	#Used to store the error over time
	error = []

	#List of batches
	batches = []

	#The number of batches
	num_batches = math.ceil(len(train_set)/batchsize)

	used_rounds = 0

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
		used_rounds = r+1
	sys.stdout.write('\r')
	model['training_instances_used'] = used_rounds * len(train_set)
	return model

def calculateError(db, output_name):
	error = 0
	for instance in db:
		target = instance[output_name]
		output = instance['output']
		if((output> 0.5 and float(target) < 0.5) or (output < 0.5 and float(target) > 0.5)):
			error += 1
	return error/len(db)



#Classifies a test set of instances using the NN model
def classifySlow(model, test_set, output_name, show_output, act_type=0):
	if(show_output): print('\nclassifying')
	correct = 0

	classified_instances = []

	for instance in test_set:
		X, target = NN.splitIO([instance], output_name)
		l0 = np.hstack((np.ones((X.shape[0], 1)), X))
		l1 = np.hstack((np.ones((X.shape[0], 1)), NN.activation_func(np.dot(l0, model['W_hidden_1']), type=act_type)))
		l2 = np.hstack((np.ones((X.shape[0], 1)), NN.activation_func(np.dot(l1, model['W_hidden_2']), type=act_type)))
		l3 = np.hstack((np.ones((X.shape[0], 1)), NN.activation_func(np.dot(l2, model['W_hidden_3']), type=act_type)))
		output = NN.activation_func(np.dot(l3, model['W_output']), type=act_type)

		if(show_output): print('target:', target, 'output:',output[0])
		instance_copy = instance.copy()
		instance_copy['output'] = output[0]
		classified_instances.append(instance_copy)

		if((output[0]> 0.5 and float(target) > 0.5) or (output[0] < 0.5 and float(target) < 0.5)):
			correct += 1

	if show_output:
		print('Accuracy (3 layers): ', repr(round(100*correct/len(test_set),2)))
	return classified_instances

#Classifies the network much faster
def classify(model, test_set, output_name, show_output, act_type=0):
	if(show_output): print('\nclassifying')
	correct = 0

	classified_instances = []

	X, target = NN.splitIO(test_set, output_name)
	l0 = np.hstack((np.ones((X.shape[0], 1)), X))
	l1 = np.hstack((np.ones((X.shape[0], 1)), NN.activation_func(np.dot(l0, model['W_hidden_1']), type=act_type)))
	l2 = np.hstack((np.ones((X.shape[0], 1)), NN.activation_func(np.dot(l1, model['W_hidden_2']), type=act_type)))
	l3 = np.hstack((np.ones((X.shape[0], 1)), NN.activation_func(np.dot(l2, model['W_hidden_3']), type=act_type)))
	output = NN.activation_func(np.dot(l3, model['W_output']), type=act_type)

	for idx, instance in enumerate(test_set):
		cur_target = target[idx]
		cur_output = output[idx]

		if(show_output): print('target:', cur_target, 'output:',cur_output[0])
		instance_copy = instance.copy()
		instance_copy['output'] = cur_output#[0]
		classified_instances.append(instance_copy)

		if((cur_output[0]> 0.5 and float(cur_target) > 0.5) or (cur_output[0] < 0.5 and float(cur_target) < 0.5)):
			correct += 1

	if show_output:
		print('Accuracy (3 layers): ', repr(round(100*correct/len(test_set),2)))
	return classified_instances

