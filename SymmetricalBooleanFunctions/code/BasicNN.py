#Basic neural network
#Author: Cor Steging
####

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import math
import sys
import copy

#Reads a dataset from CSV
def readCSV(path):
	with open(path) as input_file:
		reader = csv.DictReader(input_file)
		db = [row for row in reader]
	#Sort the keys of each 
	new_db = []
	keys = list(db[0].keys())
	keys.sort()
	for row in db:
		instance = {}
		for key in keys:
			instance[key] = row[key]
		new_db.append(instance)
	return new_db 

#Exports the model to files
def exportModel(model, filename):
	file = open("models/"+filename+".txt", "w")
	num_hidden_layers = model["num_hidden"]
	file.write(repr(num_hidden_layers)+'\n')
	if(num_hidden_layers == 1):
		np.savetxt('models/model_1L_W_1.out', model["W_hidden"])
		np.savetxt('models/model_1L_W_output.out', model["W_output"])
	if(num_hidden_layers == 2):
		np.savetxt('models/model_2L_W_1.out', model["W_hidden_1"])
		np.savetxt('models/model_2L_W_2.out', model["W_hidden_2"])
		np.savetxt('models/model_2L_W_output.out', model["W_output"])
	if(num_hidden_layers == 3):
		np.savetxt('models/model_3L_W_1.out', model["W_hidden_1"])
		np.savetxt('models/model_3L_W_2.out', model["W_hidden_2"])
		np.savetxt('models/model_3L_W_3.out', model["W_hidden_3"])
		np.savetxt('models/model_3L_W_output.out', model["W_output"])
	for f in model['features']:
		file.write(f + ', ')
	file.close() 

#imports the model from files
def importModel(path):
	file = open(path + "model_info.txt", "r")
	model = {}
	data = file.read()
	num_hidden_layers = int(data[0])
	model['num_hidden'] = num_hidden_layers
	if(num_hidden_layers == 1):
		model["W_hidden"] = np.genfromtxt(path + "model_1L_W_1.out", dtype='float')
		model["W_output"] = np.genfromtxt(path + "model_1L_W_output.out", dtype='float')
	if(num_hidden_layers == 2):
		model["W_hidden_1"] = np.genfromtxt(path + "model_2L_W_1.out", dtype='float')
		model["W_hidden_2"] = np.genfromtxt(path + "model_2L_W_2.out", dtype='float')
		model["W_output"] = np.genfromtxt(path + "model_2L_W_output.out", dtype='float')
	if(num_hidden_layers == 3):
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


#The default sigmoid function
def sigmoid(x, deriv=False):
	if deriv is True:
		return (x*(1-x))
	return 1/(1+np.exp(-x))

#The hyperbolic tangent function
def tanH(x, deriv=False):
	if deriv is True:
		x[x<=0] = 0
		x[x>0] = 1
		return 1/((x*x)+1)
	return (2/(1+np.exp(-2*x)))-1

#The rectified linear unit function (ReLU)
def ReLU(x, deriv=False):
	if deriv is True:
		x[x<=0] = 0
		x[x>0] = 1
		return x
	return np.maximum(0,x)

#The leaky ReLU function
def Leaky_ReLU(x, deriv=False):
	if deriv is True:
		x[x<=0] = 0.01
		x[x>0] = 1
		return x
	x[x<=0] = 0.01*x
	x[x>0] = x
	return x

def plotErrorRate(error, plot, name='error_rate'):
	fig = plt.gcf()
	plt.figure()
	fig.tight_layout()
	plt.plot(error)
	plt.ylabel('Error rate')
	plt.xlabel('Epochs')
	fig.tight_layout()
	plt.savefig('../models/errors/'+name+'.png')
	pd.DataFrame(error).to_csv('../models/errors/'+name+'.csv', index=False, header=False)
	if(plot): plt.show()


#The general activation function, used to select a specific function
def activation_func(x, deriv=False, type=0):
	if type == 0: return sigmoid(x, deriv)
	elif type == 1: return tanH(x, deriv)
	elif type == 2: return ReLU(x, deriv)
	elif type == 3: return Leaky_ReLU(x, deriv)
	return sigmoid(x, deriv)


#Prints information about the model
def printModel(model):
	print('Hidden weights:', model['W_hidden'].shape)
	print(model['W_hidden'])
	print('Output weights:', model['W_output'].shape)
	print(model['W_output'].T)
	print(model['features'])

def printDBInfo(db):
	print(repr(len(db)) + ' instances with ' + repr(len(list(db[0].keys()))) + ' features')

def minimizeWeights(model):
	max_val = 0
	min_val = 100
	for weight in np.nditer(model['W_hidden']):
		if weight > max_val:
			max_val = weight
		if weight < min_val:
			min_val = weight
	for weight in np.nditer(model['W_output']):
		if weight > max_val:
			max_val = weight
		if weight < min_val:
			min_val = weight
	
	new_hidden_weights = copy.deepcopy(model['W_hidden'])
	new_output_weights = copy.deepcopy(model['W_output'])


	for weight in np.nditer(model['W_hidden']):
		weight = 2*(weight-min_val)/(max_val - min_val) - 1
	for weight in np.nditer(model['W_output']):
		weight = 2*(weight-min_val)/(max_val - min_val) - 1

	print(min_val, max_val)
	
	return model


def pruneModel(model, n2):
	for weight_id, weight in enumerate(model['W_hidden'][0]):
		values = []
		for output_weight in model['W_output'][0]:
			values.append(output_weight * weight)
		print(max(values))
		if max(values) <= 4*n2:
			np.delete(model['W_hidden'], weight_id, 0)

	for weight_id, output_weight in enumerate(model['W_output']):
		if output_weight <= 4*n2:
			np.delete(model['W_output'], weight_id, 0)
	return model


#Creates a neural network model
def createModel(num_input_nodes, num_hidden_layers, num_hidden_nodes, num_output_nodes, learning_rate, features, output_name):

	np.random.seed(0)

	#Create weights for the hidden layer nodes (+1 for the biases)
	W_hidden = 2*np.random.random((num_input_nodes + 1, num_hidden_nodes[0])) - 1

	#Weights for hidden layer (+1 for the biases)
	W_output = 2*np.random.random((num_hidden_nodes[0] + 1, num_output_nodes)) - 1

	features.remove(output_name)

	#Return the model as a dictionary of the weights
	return {"num_hidden": num_hidden_layers, "W_hidden":W_hidden, "W_output":W_output, "learning_rate":learning_rate, "features":features}


#Splits a list of dictionaries into an input and output numpy matrix
def splitIO(db, output_name):
	#print('splitting')
	df = pd.DataFrame(db)
	y = df[[output_name]].as_matrix()
	del df[output_name]
	x = df.as_matrix()
	return np.asfarray(x),np.asfarray(y)


def backprop(model, X, target, act_type):
	#Forward propagate
	input_layer_outputs = np.hstack((np.ones((X.shape[0], 1)), X))
	hidden_layer_outputs = np.hstack((np.ones((X.shape[0], 1)), activation_func(np.dot(input_layer_outputs, model['W_hidden']), type=act_type)))
	output_layer_outputs = activation_func(np.dot(hidden_layer_outputs, model['W_output']), type=act_type)

	#Compute the errors
	output_error = output_layer_outputs - target
	local_error = sum(np.absolute(output_error))
	#hidden_error = hidden_layer_outputs[:, 1:] * (1 - hidden_layer_outputs[:, 1:]) * np.dot(output_error, model['W_output'].T[:, 1:])
	hidden_error = activation_func(hidden_layer_outputs[:, 1:], deriv=True, type=act_type) * np.dot(output_error, model['W_output'].T[:, 1:])

	# partial derivatives
	hidden_pd = input_layer_outputs[:, :, np.newaxis] * hidden_error[: , np.newaxis, :]
	output_pd = hidden_layer_outputs[:, :, np.newaxis] * output_error[:, np.newaxis, :]

	# average for total gradients
	total_hidden_gradient = np.average(hidden_pd, axis=0)
	total_output_gradient = np.average(output_pd, axis=0)

	#update weights
	model['W_output'] += total_output_gradient * -model["learning_rate"]
	model['W_hidden']  += total_hidden_gradient * -model["learning_rate"]
	return local_error


#Trains the model using the default batch system
def trainModelBatch(model, train_set, output_name, act_type, rounds, plot=False):
	#Prepare the dataset by splitting the input from the output
	all_x, all_y = splitIO(train_set, output_name)
	
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
	plotErrorRate(error, plot, name='1_layer_batch')
	return model

#Trains the NN model using stochastic descent
def trainModelStochastic(model, train_set, output_name, act_type, rounds, plot=False):
	#Prepare the dataset by splitting the input from the output
	all_x, all_y = splitIO(train_set, output_name)
	
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
	plotErrorRate(error, plot, name='1_layer_stochastic')
	return model

#Trains the model using the mini batch approach
def trainModelMiniBatch1(model, train_set, batchsize, output_name, act_type, rounds, plot=False):
	#Prepare the dataset by splitting the input from the output
	all_x, all_y = splitIO(train_set, output_name)
	
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
	plotErrorRate(error, plot, name='1_layer_minibatch')
	return model

#Trains the model using the mini batch approach
def trainModelMiniBatch(model, train_set, batchsize, output_name, act_type, rounds, plot=False):
	#Prepare the dataset by splitting the input from the output
	all_x, all_y = splitIO(train_set, output_name)
	
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
	plotErrorRate(error, plot, name='1_layer_minibatch')
	return model

#Classifies a test set of instances using the NN model
def classifySlow(model, test_set, output_name, show_output, act_type=0):
	#print('\nclassifying')
	correct = 0
	classified_instances = []
	for instance in test_set:
		X, target = splitIO([instance], output_name)
		l0 = np.hstack((np.ones((X.shape[0], 1)), X))
		l1 = np.hstack((np.ones((X.shape[0], 1)), activation_func(np.dot(l0, model['W_hidden']), type=act_type)))
		output = activation_func(np.dot(l1, model['W_output']), type=act_type)

		instance_copy = instance.copy()
		instance_copy['output'] = output[0][0]
		classified_instances.append(instance_copy)

		if(show_output): print('target:', target, 'output:',output[0])

		center = 0.5
		if((output[0]> center and float(target) > center) or(output[0] < center and float(target) < center)):
			correct += 1
			
	return classified_instances

#Classifies a test set of instances using the NN model
def classify(model, test_set, output_name, show_output, act_type=0):
	#print('\nclassifying')
	correct = 0
	classified_instances = []
	X, target = splitIO(test_set, output_name)
	l0 = np.hstack((np.ones((X.shape[0], 1)), X))
	l1 = np.hstack((np.ones((X.shape[0], 1)), activation_func(np.dot(l0, model['W_hidden']), type=act_type)))
	output = activation_func(np.dot(l1, model['W_output']), type=act_type)

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
