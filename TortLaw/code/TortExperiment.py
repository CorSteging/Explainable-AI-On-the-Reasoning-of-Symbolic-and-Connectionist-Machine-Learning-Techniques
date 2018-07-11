#Calculates how well a machine learning system can learn a set of conditions
#Author: Cor Steging

import csv
import random
import pickle
import os
import ThreeLayeredNN as NN3
import TwoLayeredNN as NN2
import BasicNN as NN1
import DecisionTree as DT
from DecisionTree import Node, Leaf, Question
import GenerateTortSet as GDB
import deepRED as RED

#Reads a database as a list of dictionaries
def readCSV(path):
	with open(path) as input_file:
		reader = csv.DictReader(input_file)
		db = [row for row in reader]
	return db 

def appendResultCSV(accuracies, params, filename):
	output_instance = {}
	systems = ['DT','NN']

	output_instance['num_layers'] = params['num_hidden_layers']
	output_instance['num_nodes'] = params['num_hidden_nodes']
	for system_id, system in enumerate(accuracies):
		for idx, accuracy in enumerate(system):
			if idx == 0:
				test_set_name = 'general'
			else:
				test_set_name = 'unique'
			output_instance[systems[system_id] + '_' + test_set_name+'_accuracy'] = accuracy

	output_instance['db_size'] = params['num_train_instances']

	header = True if os.path.exists(filename) else False
	keys = output_instance.keys()
	with open(filename, 'a', newline='') as output:
		dict_writer = csv.DictWriter(output, keys, delimiter=',')
		if not header:
			dict_writer.writeheader()
		dict_writer.writerows([output_instance])


#Writes the results of the classifications to a TXT file
def exportResults(accuracies, filename):
	file = open(filename, "w")
	for system_id, system in enumerate(accuracies):
		file.write('### Decision Tree performance:\n') if system_id == 0 else file.write('\n### Neural Network performance:\n')
		for idx, accuracy in enumerate(system):
			if idx == 0:
				file.write('Accuracy on general test set: ' + repr(round(accuracy,2)) + '\n')
			if idx == 1:
				file.write('Accuracy on unique test set: ' + repr(round(accuracy,2)) + '\n')
	file.close()

def createDirs(experiment_name):
	if not os.path.exists('../models/'+experiment_name):
		os.makedirs('../models/'+experiment_name)
		os.makedirs('../models/'+experiment_name+'/NN')
		os.makedirs('../models/'+experiment_name+'/DT')
	if not os.path.exists('../outputs/'+experiment_name):
		os.makedirs('../outputs/'+experiment_name)
		os.makedirs('../outputs/'+experiment_name+'/NN')
		os.makedirs('../outputs/'+experiment_name+'/DT')
	if not os.path.exists('../results/'+experiment_name):
		os.makedirs('../results/'+experiment_name)

#Classifies and exports the outputs of NN's
def NNClassifyAndExport(model, test_set, params, path):
	if params['num_hidden_layers'] == 1:
		NN1.exportOutputs(NN1.classify(model, test_set, params['output_name'], False), path)
	if params['num_hidden_layers'] == 2:
		NN1.exportOutputs(NN2.classify(model, test_set, params['output_name'], False), path)
	if params['num_hidden_layers'] == 3:
		NN3.exportOutputs(NN3.classify(model, test_set, params['output_name'], False), path)

#Creates and trains a NN
def trainNN(db, params):
	plot = False
	show_output = False

	random.shuffle(db)
	test_set = db
	train_set = db

	num_input_nodes = len(db[0])-1
	num_output_nodes = 1

	if params['num_hidden_layers'] == 1:
		model = NN1.createModel(num_input_nodes, params['num_hidden_layers'], params['num_hidden_nodes'], num_output_nodes, params['learning_rate'], list(db[0].keys()), params['output_name'])
		model = NN1.trainModelMiniBatch(model, train_set, params['batchSize'], params['output_name'], params['act_type'], params['rounds'], plot=plot)
	if params['num_hidden_layers'] == 2:
		model = NN2.createModel(num_input_nodes, params['num_hidden_layers'], params['num_hidden_nodes'], num_output_nodes, params['learning_rate'], list(db[0].keys()), params['output_name'])
		model = NN2.trainModelMiniBatch(model, train_set, params['batchSize'], params['output_name'], params['act_type'], params['rounds'], plot=plot)
	if params['num_hidden_layers'] == 3:
		model = NN3.createModel(num_input_nodes, params['num_hidden_layers'], params['num_hidden_nodes'], num_output_nodes, params['learning_rate'], list(db[0].keys()), params['output_name'])
		model = NN3.trainModelMiniBatch(model, train_set, params['batchSize'], params['output_name'], params['act_type'], params['rounds'], plot=plot)

	if params['saveModels']:
		NN3.exportModel(model, "../models/"+params['experiment_name']+'/NN/')

	rules = None
	if params['deepRED'] == True:
		rules = RED.getRules(model, db, params['output_name'], simplify=True)
		file = open('../models/'+ params['experiment_name'] + '/NN/NN_tree.txt', "w")
		DT.getTXTReprentation(rules, file)
		file.close()
		file = open('../models/'+ params['experiment_name'] + '/NN/NN_rules.txt', "w")
		DT.getRuleReprentation(rules, file)
		file.close()

	return model, rules

#Creates a DT
def trainDT(db, params):
	features = list(db[0].keys())
	random.shuffle(db)
	output_col = DT.findOutputCol(db, params['output_name'])
	db = DT.toNumpyArray(db)
	tree = DT.buildTree(db, output_col, features, gain_level = params['gain_level'])
	DT.exportTree(tree, '../models/'+ params['experiment_name'] + '/DT/tree.out')
	file = open('../models/'+ params['experiment_name'] + '/DT/tree.txt', "w")
	DT.getTXTReprentation(tree, file)
	file.close()
	file = open('../models/'+ params['experiment_name'] + '/DT/rules.txt', "w")
	DT.getRuleReprentation(tree, file)
	file.close()
	return tree

#Prints and returns the accuracies of the systems on the test sets
def printAccuracies(outputs, output_name):
	accuracies = []
	for output_id, output in enumerate(outputs):
		current_accuracies = []
		print('### Decision Tree performance:') if output_id == 0 else print('\n### Neural Network performance:')
		if output_id == 2: 
			print("\t (Using deepRED rules)")
		for idx, test_set in enumerate(output):
			counter = 0
			for instance in test_set:
				if (float(instance[output_name]) >= 0.5 and float(instance['output'])) >= 0.5 or (float(instance[output_name]) < 0.5 and float(instance['output']) < 0.5):
					counter += 1
			accuracy = 100 * counter/len(test_set)
			current_accuracies.append(accuracy)
			if idx == 0:
				print('Accuracy on general test set: ' + repr(round(accuracy,2)))
			if idx == 1:
				print('Accuracy on unique test set: ' + repr(round(accuracy,2)))
		accuracies.append(current_accuracies)
	return accuracies

#Runs the entire experiment
def runMain(params):
	#Generate the datasets
	GDB.generateDatasets(params['num_train_instances'], params['num_test_instances'], params['output_ratio'], params['experiment_name'], params['noise_level'])

	#Make folders if they don't exist already
	experiment_name = params['experiment_name']
	output_name = params['output_name']
	createDirs(experiment_name)

	#The training db
	train_db = readCSV('../datasets/' + experiment_name + '/train_db.csv')

	#The general test db
	general_test_db = readCSV('../datasets/' + experiment_name + '/test_db.csv')

	#The unique test db
	unique_test_db = readCSV('../datasets/' + experiment_name + '/unique_test_db.csv')

	#The Decision tree and Neural Network model
	if params['trainFirst'] == False and os.path.exists('../models/DT/' +experiment_name+'/tree.out') and os.path.exists('../models/NN/' +experiment_name+'/model_info.txt'):
		DT_model = DT.importTree('../models/' +experiment_name+'/DT/tree.out')
		NN_model = NN1.importModel("../models/"+experiment_name+'/NN/')
	else:
		print('# Building Decision Tree')
		DT_model = trainDT(train_db, params)
		DT_model = DT.importTree('../models/' +experiment_name+'/DT/tree.out')
		print('# Training Neural Network')
		NN_model, NN_rules = trainNN(train_db, params)

	#classify the test sets using both DT and NN
	if params['classifyTestSets'] == True:
		print('# Classifying')
	
		#The general test set
		DT.exportOutputs(DT.classifySet(DT.toNumpyArray(general_test_db), DT_model, DT.findOutputCol(train_db, output_name), show_output = False), '../outputs/' + experiment_name + '/DT/output.csv', list(train_db[0].keys()))
		NNClassifyAndExport(NN_model, general_test_db, params, '../outputs/' + experiment_name + '/NN/output.csv')
		if params['deepRED'] == True:
			DT.exportOutputs(DT.classifySet(DT.toNumpyArray(general_test_db), NN_rules, DT.findOutputCol(train_db, output_name), show_output = False), '../outputs/' + experiment_name + '/NN/rules_output.csv', list(train_db[0].keys()))

		DT.exportOutputs(DT.classifySet(DT.toNumpyArray(unique_test_db), DT_model, DT.findOutputCol(train_db, output_name), show_output = False), '../outputs/' + experiment_name + '/DT/unique_output.csv', list(train_db[0].keys()))
		NNClassifyAndExport(NN_model, unique_test_db, params, '../outputs/' + experiment_name + '/NN/unique_output.csv')
		if params['deepRED'] == True:
			DT.exportOutputs(DT.classifySet(DT.toNumpyArray(unique_test_db), NN_rules, DT.findOutputCol(train_db, output_name), show_output = False), '../outputs/' + experiment_name + '/NN/rules_unique_output.csv', list(train_db[0].keys()))

	#The output files of the DT
	print('# Reading outputs')
	DT_outputs = [readCSV('../outputs/' + experiment_name + '/DT/output.csv')]
	DT_outputs.append(readCSV('../outputs/' + experiment_name + '/DT/unique_output.csv'))

	#The output files of the NN
	NN_outputs = [readCSV('../outputs/' + experiment_name + '/NN/output.csv')]
	NN_outputs.append(readCSV('../outputs/' + experiment_name + '/NN/unique_output.csv'))
	NN_rules_outputs = []
	if params['deepRED'] == True:
		NN_rules_outputs = [readCSV('../outputs/' + experiment_name + '/NN/rules_output.csv')]
		NN_rules_outputs.append(readCSV('../outputs/' + experiment_name + '/NN/rules_unique_output.csv'))
	
	accuracies = printAccuracies([DT_outputs, NN_outputs, NN_rules_outputs], output_name)
	exportResults(accuracies, '../results/' + experiment_name + '/results.txt')
	appendResultCSV(accuracies, params, '../results/' + experiment_name + '/all_results.csv')

if __name__ == "__main__":

	##### VARIABLES ######

	params = {}

	#The name of the experiment
	params['experiment_name'] = 'Tort_experiment_noise'

	#Number of instances of the train set
	params['num_train_instances'] = 3000#150000

	#Number of instances of the test set
	params['num_test_instances'] = 2000#150000

	#Ratio of output variable (true to false)
	params['output_ratio'] = 0.5

	#The name of the dependent variable
	params['output_name'] = 'dut'
	
	#The name of the dependent variable
	params['noise_level'] = 0.0

	
	### Decision Tree Options ###
	
	#The level of gain in the creation process
	params['gain_level'] = 0

	
	### Neural Network Options ###
	
	#The type of activation function used
	params['act_type'] = 0
	
	#The number of hidden layers (1,2 or 3)
	params['num_hidden_layers'] = 3

	#The number of hidden nodes for each layer
	params['num_hidden_nodes'] = [25, 10, 3]
	
	#The number of training rounds
	params['rounds'] = 2000
	
	#The learning rate of the network
	params['learning_rate'] = 0.05

	#The Batch size
	params['batchSize'] = 50#round(params['num_train_instances']/5000)  #50

	#Should deepRED be used?
	params['deepRED'] = False#True


	### ETC ###

	#Should the NN and DT be trained first?
	params['trainFirst'] = True

	#Should the test sets be classified?
	params['classifyTestSets'] = True

	#Should the models be saved?
	params['saveModels'] = True

	#########################
	
	runMain(params)



