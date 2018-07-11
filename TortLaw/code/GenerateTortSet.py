#Generates a dataset using the rules from Dutch tort law
#Author: Cor Steging

import csv
import random
import sys
import string
import math
import pickle
import os

#Exports the database to a csv file
def exportCSV(database, filename):
	keys = database[0].keys()

	with open(filename, 'w', newline='') as output:
		dict_writer = csv.DictWriter(output, keys, delimiter=',')
		dict_writer.writeheader()
		dict_writer.writerows(database)

def isTrue(instance):
	unl = 0
	imp = 0
	dut = 0

	if instance['vun'] == 1 or (instance['vst'] == 1 and instance['jus'] == 0) or (instance['vrt'] == 1 and instance['jus'] == 0):
		unl = 1

	if instance['ico'] == 1 or instance['ila'] == 1 or instance['ift'] == 1:
		imp = 1

	if instance['dmg'] == 1 and unl == 1 and imp == 1 and instance['cau'] == 1 and not (instance['vst']==1 and instance['prp'] == 0):
		return True

	return False


#Do we represent unl en imp? They are the intermediate states
def generatePossibleValues():
	possible_values = []
	N = 10 #how many variables, each can have 0 or 1 #There are 13 variables including dut, imp and unl
	num_combinations = int(math.pow(2, N)) #How many possible combinations can exist

	#Find the binary representation of each possibility, which can be used as values for each variable
	for x in range(0,num_combinations):
		binary = ('000000000000000000000000000000' + str(bin(x))[2:])[-N:]
		possible_value = []
		for character in binary:
			possible_value.append(int(character))
		possible_values.append(possible_value)

	names_all = ['dut', 'dmg', 'unl', 'imp', 'cau', 'vrt', 'vst', 'vun', 'ift', 'ila', 'ico', 'jus', 'prp']
	names = ['dmg', 'cau', 'vrt', 'vst', 'vun', 'ift', 'ila', 'ico', 'jus', 'prp']

	true_values = []
	false_values = []

	for value in possible_values:
		
		instance = {}
		for val_id, val in enumerate(value):
			instance[names[val_id]] = val

		if isTrue(instance):
			true_values.append(value)
		else:
			false_values.append(value)

	return [possible_values, true_values, false_values]


def createTestDB(num_instances, output_ratio, possible_values):
	db = []
	output = 1
	for instance_id in range(0, num_instances):
		if instance_id > num_instances * output_ratio:
			output = 0
		instance = createInstance(instance_id, output, possible_values)
		db.append(instance)
	return db

def createInstance(instance_id, output, possible_values):
	names = ['dmg', 'cau', 'vrt', 'vst', 'vun', 'ift', 'ila', 'ico', 'jus', 'prp']
	
	if output == 1:
		values = possible_values[1][random.randint(0, len(possible_values[1])-1)]
	else:
		values = possible_values[2][random.randint(0, len(possible_values[2])-1)]

	instance = {}
	for val_id, val in enumerate(values):
		instance[names[val_id]] = val

	if isTrue(instance):
		instance['dut'] = 1
	else:
		instance['dut'] = 0

	return instance


def createDB(num_instances, output_ratio, possible_values):
	db = []
	output = 1
	for instance_id in range(0, num_instances):
		if instance_id > num_instances * output_ratio:
			output = 0
		instance = createInstance(instance_id, output, possible_values)
		db.append(instance)
	return db

#Creates a db consisting of all possible value combinations
def createUniqueDB(possible_values):
	names = ['dmg', 'cau', 'vrt', 'vst', 'vun', 'ift', 'ila', 'ico', 'jus', 'prp']
	db = []
	for values in possible_values[0]:
		instance = {}
		for val_id, val in enumerate(values):
			instance[names[val_id]] = val
		if isTrue(instance):
			instance['dut'] = 1
		else:
			instance['dut'] = 0
		db.append(instance)
	return db

def addBooleanNoise(db, noise_level = 0.1):
	noise_counter = 0
	new_db = []
	output_name = 'dut'
	for instance_id, instance in enumerate(db):
		if random.uniform(0, 1) <= noise_level:
			#add noise (invert the value of one feature, but not the output value)
			noise_counter += 1
			noisy_feature = random.randint(0,len(instance)-2)
			if list(instance.keys())[noisy_feature] == output_name: noisy_feature += 1
			noisy_feature = list(instance.keys())[noisy_feature]
			instance[noisy_feature] = 1 - instance[noisy_feature]
	print(repr(noise_counter) + ' instances with noise: ' + repr(100*noise_counter/len(db)) + ' percent')	
	findMistakes(db)
	return db

#Finds how many instances have an incorrect output label after adding noise
def findMistakes(db):
	output_name = 'dut'
	cor_counter = 0
	incor_counter = 0
	for instance in db:
		if (isTrue(instance) and instance[output_name] == 1) or  (not isTrue(instance) and instance[output_name] == 0):
			cor_counter += 1
		else:
			incor_counter += 1
	print('out of ' + repr(len(db)) + ' instance, ' + repr(cor_counter) + ' instances are correct and '+ repr(incor_counter) + ' instances are incorrect, which is ' + repr(100*incor_counter/len(db)) + ' percent incorrect')



#There are 13 variables
def generateDatasets(num_train_instances, num_test_instances, output_ratio, experiment_name, noise_level=0):

	if not os.path.exists('../datasets/' + experiment_name):
		os.makedirs('../datasets/' + experiment_name)

	possible_values = generatePossibleValues()
	db = createDB(num_train_instances, output_ratio, possible_values)
	if noise_level: db = addBooleanNoise(db, noise_level)
	exportCSV(db, '../datasets/' + experiment_name + '/train_db.csv')
	test_db = createDB(num_test_instances, output_ratio, possible_values)
	if noise_level: test_db = addBooleanNoise(test_db, noise_level)
	exportCSV(test_db, '../datasets/' + experiment_name + '/test_db.csv')
	unique_test_db = createUniqueDB(possible_values)
	exportCSV(unique_test_db, '../datasets/' + experiment_name + '/unique_test_db.csv')

if __name__ == "__main__":

	#Variables used:
	################

	#Number of instances of the train set
	num_train_instances = 150000

	#Number of instances of the test set
	num_test_instances = 150000

	#Ratio of output variable (true to false)
	output_ratio = 0.5

	#The name of the experiment
	experiment_name = 'Tort_experiment_noise'

	generateDatasets(num_train_instances, num_test_instances, output_ratio, experiment_name, 0.1)


