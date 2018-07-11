import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import math
import sys
import random

import BasicNN as NN1
import TwoLayeredNN as NN2
import ThreeLayeredNN as NN3



def oneLayer():

	model = NN1.createModel(num_input_nodes, num_hidden_layers, num_hidden_nodes, num_output_nodes, learning_rate, list(db[0].keys()), output_name)
	if(gd_type==1): model = NN1.trainModelBatch(model, train_set, output_name, act_type, rounds, plot=plot)
	if(gd_type==2): model = NN1.trainModelStochastic(model, train_set, output_name, act_type, rounds, plot=plot)
	if(gd_type==3): model = NN1.trainModelMiniBatch(model, train_set, 50, output_name, act_type, rounds, plot=plot)
	NN1.exportModel(model, "model1layer")
	return NN1.classify(model, test_set, output_name, show_output)

def twoLayers():

	model = NN2.createModel(num_input_nodes, num_hidden_layers, num_hidden_nodes, num_output_nodes, learning_rate, list(db[0].keys()), output_name)
	if(gd_type==1): model = NN2.trainModelBatch(model, train_set, output_name, act_type, rounds, plot=plot)
	if(gd_type==2): model = NN2.trainModelStochastic(model, train_set, output_name, act_type, rounds, plot=plot)
	if(gd_type==3): model = NN2.trainModelMiniBatch(model, train_set, 50, output_name, act_type, rounds, plot=plot)
	NN1.exportModel(model, "model2layer")
	return NN2.classify(model, test_set, output_name, show_output)

def threeLayers():

	model = NN3.createModel(num_input_nodes, num_hidden_layers, num_hidden_nodes, num_output_nodes, learning_rate, list(db[0].keys()), output_name)
	if(gd_type==1): model = NN3.trainModelBatch(model, train_set, output_name, act_type, rounds, plot=plot)
	if(gd_type==2): model = NN3.trainModelStochastic(model, train_set, output_name, act_type, rounds, plot=plot)
	if(gd_type==3): model = NN3.trainModelMiniBatch(model, train_set, 50, output_name, act_type, rounds, plot=plot)
	NN1.exportModel(model, "model3layer")
	return NN3.classify(model, test_set, output_name, show_output)


if __name__ == "__main__":

	###########################
	#SET THE DATASET PARAMETERS
	############################


	#Read in the database and divide it into a test and train set
	db = NN1.readCSV("../datasets/welfare.csv")

	#The name of the output attribute
	output_name = "satisfied"

	db = NN1.readCSV("../datasets/prepared_welfare.csv")
	age_db = NN1.readCSV("../datasets/prepared_ageExperiment.csv")
	dist_db = NN1.readCSV("../datasets/prepared_distExperiment.csv")
	test_db = NN1.readCSV("../datasets/prepared_welfareTest.csv")
	oneFail_db = NN1.readCSV("../datasets/prepared_welfare_onefail.csv")
	oneFailtest_db = NN1.readCSV("../datasets/prepared_welfareTest_onefail.csv")


	test_set = test_db
	train_set = db

	random.shuffle(test_set)
	random.shuffle(train_set)

	NN1.printDBInfo(train_set)
	NN1.printDBInfo(test_set)

	####################
	#GENERAL PARAMETERS
	####################

	#Type of activation function
	act_type = 0

	#Type of gradient descent (1 = Batch, 2 = stochastic, 3 = mini-batch)
	gd_type = 2

	#Proportion test- to training-set size
	test_prop = 0.2

	#Number of input nodes
	num_input_nodes = len(db[0])-1

	#The number of output nodes
	num_output_nodes = 1

	#Learning rate
	learning_rate = 0.05

	#number of training rounds
	rounds = 5000

	#Whether or not to plot the learning graph
	plot = False

	#Whether or not the output of the classifier is shown
	show_output = False


	#########################
	# 1 layer NN
	#########################
	print('### 1 Layer ###')

	num_hidden_layers = 1
	num_hidden_nodes = 12
	outputs_1 = oneLayer()

	model = NN1.importModel("models/model1layer.txt")

	print("\n1 test db A")
	test_set = test_db
	print(len(test_set[0]))
	outputs_1 = NN1.classify(model, test_set, output_name, show_output)
	NN1.exportOutputs(outputs_1, 'Outputs/A1.csv')

	print("\n1 test db B")
	test_set = oneFailtest_db
	print(len(test_set[0]))
	outputs_1 = NN1.classify(model, test_set, output_name, show_output)
	NN1.exportOutputs(outputs_1, 'Outputs/B1.csv')

	print("\n1 dist db")
	test_set = dist_db
	print(len(test_set[0]))
	outputs_1 = NN1.classify(model, test_set, output_name, show_output)
	NN1.exportOutputs(outputs_1, 'Outputs/dist1.csv')

	print("\n1 age db")
	test_set = age_db
	print(len(test_set[0]))
	outputs_1 = NN1.classify(model, test_set, output_name, show_output)
	NN1.exportOutputs(outputs_1, 'Outputs/age1.csv')


	#########################
	# 2 layer NN
	#########################
	print('### 2 Layer ###')
	test_set = db
	num_hidden_layers = 2
	num_hidden_nodes = [24, 6]
	outputs_2 = twoLayers()

	model = NN1.importModel("models/model2layer.txt")

	print("\n2 test db A")
	test_set = test_db
	outputs_2 = NN2.classify(model, test_set, output_name, show_output)
	NN1.exportOutputs(outputs_2, 'Outputs/Noise/A2.csv')

	print("\n2 test db B")
	test_set = oneFailtest_db
	outputs_2 = NN2.classify(model, test_set, output_name, show_output)
	NN1.exportOutputs(outputs_2, 'Outputs/Noise/B2.csv')

	print("\n2 dist db")
	test_set = dist_db
	outputs_2 = NN2.classify(model, test_set, output_name, show_output)
	NN1.exportOutputs(outputs_2, 'Outputs/Noise/dist2.csv')

	print("\n2 age db")
	test_set = age_db
	outputs_2 = NN2.classify(model, test_set, output_name, show_output)
	NN1.exportOutputs(outputs_2, 'Outputs/Noise/age2.csv')


	#########################
	# 3 layer NN
	#########################
	print('### 3 Layer ###')
	test_set = db
	num_hidden_layers = 3
	num_hidden_nodes = [24, 10, 3]
	outputs_3 = threeLayers()

	model = NN1.importModel("models/model3layer.txt")

	print("\n3 test db A ")
	test_set = test_db
	outputs_3 = NN3.classify(model, test_set, output_name, show_output)
	NN1.exportOutputs(outputs_3, 'Outputs/Noise/A3.csv')

	print("\n3 test db B")
	test_set = oneFailtest_db
	outputs_3 = NN3.classify(model, test_set, output_name, show_output)
	NN1.exportOutputs(outputs_3, 'Outputs/Noise/B3.csv')

	print("\n3 dist db")
	test_set = dist_db
	outputs_3 = NN3.classify(model, test_set, output_name, show_output)
	NN1.exportOutputs(outputs_3, 'Outputs/Noise/dist3.csv')

	print("\n3 age db")
	test_set = age_db
	outputs_3 = NN3.classify(model, test_set, output_name, show_output)
	NN1.exportOutputs(outputs_3, 'Outputs/Noise/age3.csv')

	print("\n3 dist db")
	test_set = dist_db_noise
	outputs_3 = NN3.classify(model, test_set, output_name, show_output)
	NN1.exportOutputs(outputs_3, 'Outputs/Noise/dist_noise3.csv')

	print("\n3 age db")
	test_set = age_db_noise
	outputs_3 = NN3.classify(model, test_set, output_name, show_output)
	NN1.exportOutputs(outputs_3, 'Outputs/Noise/age_noise3.csv')


