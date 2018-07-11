#Generates a dataset using a set of rules
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
	#Sort the keys of each 
	new_db = []
	keys = list(database[0].keys())
	#print("\n", filename, ":\n", keys)
	keys.sort()
	keys = sorted(keys)
	for row in database:
		instance = {}
		for key in keys:
			instance[key] = row[key]
		new_db.append(instance)
	keys = new_db[0].keys()

	with open(filename, 'w', newline='') as output:
		dict_writer = csv.DictWriter(output, keys, delimiter=',')
		dict_writer.writeheader()
		dict_writer.writerows(new_db)

#Export the conditions
def exportConditions(conditions, filename):
	with open(filename, 'wb') as fp:
		pickle.dump(conditions, fp)

#Imports the conditions
def importConditions(filename):
	with open (filename, 'rb') as fp:
		return pickle.load(fp)

#0 Random condition
def RandomCondition(instance, output, params, current_var_id, fail, possible_values, instance_id):
	variable_name = string.ascii_uppercase[current_var_id]
	instance[variable_name] = random.randint(0, 100)/100
	current_var_id += 1
	return instance, current_var_id

	return instance, current_var_id


#1 parity function (XOR if params == 2)
def ParityFunction(instance, output, params, current_var_id, fail, possible_values, instance_id):
	variable_names = string.ascii_uppercase
	
	values = []

	if output == 1:
		#values = possible_values[1][instance_id%(len(possible_values[1]))]
		values = possible_values[1][random.randint(0, len(possible_values[1])-1)]
	if output == 0:
		#values = possible_values[2][instance_id%(len(possible_values[2]))]
		values = possible_values[2][random.randint(0, len(possible_values[2])-1)]

	for val_id, val in enumerate(values):
		instance[variable_names[current_var_id+val_id]] = val

	#print(values)
	#print(instance)
	current_var_id += params[0]

	return instance, current_var_id


#2 m-out-of-n/treshold condition: (A ^ B ^ C) V (!A ^ B ^ C) V (A ^ !B ^ C) V (A ^ B ^ !C)
def MOutOfNCondition(instance, output, params, current_var_id, fail, possible_values, instance_id):
	variable_names = string.ascii_uppercase
	m = params[1]
	
	values = []

	if output == 1:
		values = possible_values[1][random.randint(0, len(possible_values[1])-1)]
	if output == 0:
		values = possible_values[2][random.randint(0, len(possible_values[2])-1)]

	for val_id, val in enumerate(values):
		instance[variable_names[current_var_id+val_id]] = val

	current_var_id += params[0]

	return instance, current_var_id

#3 exact value function
def ExactValueFunction(instance, output, params, current_var_id, fail, possible_values, instance_id):
	variable_names = string.ascii_uppercase
	m = params[1]
	
	values = []

	if output == 1:
		values = possible_values[1][random.randint(0, len(possible_values[1])-1)]
	if output == 0:
		values = possible_values[2][random.randint(0, len(possible_values[2])-1)]

	for val_id, val in enumerate(values):
		instance[variable_names[current_var_id+val_id]] = val

	current_var_id += params[0]

	return instance, current_var_id

#3 counting function
def CountingFunction(instance, output, params, current_var_id, fail, possible_values, instance_id):
	variable_names = string.ascii_uppercase
	k = params[1]
	m = params[2]
	
	values = []

	if output == 1:
		values = possible_values[1][random.randint(0, len(possible_values[1])-1)]
	if output == 0:
		values = possible_values[2][random.randint(0, len(possible_values[2])-1)]

	for val_id, val in enumerate(values):
		instance[variable_names[current_var_id+val_id]] = val

	current_var_id += params[0]

	return instance, current_var_id



#5 nested implication: A -> B -> Z
def NestedCondition(instance, output, params, current_var_id, fail, possible_values, instance_id):
	variable_names = string.ascii_uppercase
	
	if output == 1:
		instance[variable_names[current_var_id]] = random.randint(0, 1)
		instance[variable_names[current_var_id+1]] = 1
	else:
		if fail == True:
			instance[variable_names[current_var_id]] = 0
			instance[variable_names[current_var_id+1]] = 0
		else:
			instance[variable_names[current_var_id]] = random.randint(0, 1)
			instance[variable_names[current_var_id+1]] = 1 if instance[variable_names[current_var_id]] == 1 else random.randint(0, 1)
	return instance, current_var_id+2

#Translates the integer representation of a condition into a string representation
def ConditionToName(condition):
	if condition[0] == 0: #Random
		return 'Random'
	if condition[0] == 1: #Parity
		return 'Parity function'
	if condition[0] == 2: #M-out-of-N
		return 'M-out-of-N'
	if condition[0] == 3: #Exact value
		return 'Exact value'
	if condition[0] == 4: #Counting
		return 'Exact value'
	if condition[0] == 5: #Nested implications
		return 'Nested implication'

def printConditions(conditions):
	print('Datasets are generated using the following rules:')
	for cond_id, condition in enumerate(conditions):
		params = ' '
		if condition[1] != []:
			params += repr(condition[1])
		print(repr(cond_id+1) + '. ' + ConditionToName(condition) + params)
	print('\n')

#Changes the feature names to better indicate the conditions
def changeFeatureNames(db, conditions, num_noise_vars):
	default_names = string.ascii_uppercase
	current_var_id = 0

	new_names = []
	new_db = []

	for condition in conditions:
		if condition[0] == 0: #Random
			new_names.append(default_names[current_var_id] + '_random')
			current_var_id += 1
		if condition[0] == 1: #Parity
			n = condition[1][0]
			for x in range(0,n):
				new_names.append(default_names[current_var_id + x] + '_Parity')
			current_var_id += 2
		if condition[0] == 2: #M-out-of-N
			n = condition[1][0]
			for x in range(0,n):
				new_names.append(default_names[current_var_id + x] + '_MofN')
			current_var_id += n #N
		if condition[0] == 3: #Exact value
			n = condition[1][0]
			for x in range(0,n):
				new_names.append(default_names[current_var_id + x] + '_Exact')
			current_var_id += n #N
		if condition[0] == 4: #COUNTING
			n = condition[1][0]
			for x in range(0,n):
				new_names.append(default_names[current_var_id + x] + '_Counting')
			current_var_id += n #N
		if condition[0] == 5: #Nested implication
			new_names.append(default_names[current_var_id] + 'NestedImp')
			new_names.append(default_names[current_var_id+1] + '_NestedImp')
			current_var_id += 2

	for x in range(0, num_noise_vars):
		new_names.append(default_names[current_var_id] + '_Noise')
		current_var_id += 1

	for instance in db:
		new_instance = {}
		new_instance['Z'] = instance['Z']
		del instance['Z']
		for feature_id, feature in enumerate(instance):
			new_instance[new_names[feature_id]] = instance[feature]
		new_db.append(new_instance)
	return new_db

def generatePossibleValues(condition):
	possible_values = []
	if condition[0] == 0: #RANDOM
		for A in range(0,100, 5):
			possible_values.append(A)
	if condition[0] == 1: #Parity function
		N = condition[1][0] #how many variables, each can have 0 or 1
		num_combinations = int(math.pow(2, N)) #How many possible combinations can exist

		#Find the binary representation of each possibility, which can be used as values for each variable
		for x in range(0,num_combinations):
			binary = ('000000000000000000000000000000' + str(bin(x))[2:])[-N:]
			possible_value = []
			for character in binary:
				possible_value.append(int(character))
			possible_values.append(possible_value)

	if condition[0] == 2: #M out of n
		N = condition[1][0] #how many variables, each can have 0 or 1
		num_combinations = int(math.pow(2, N)) #How many possible combinations can exist

		#Find the binary representation of each possibility, which can be used as values for each variable
		for x in range(0,num_combinations):
			binary = ('000000000000000000000000000000' + str(bin(x))[2:])[-N:]
			possible_value = []
			for character in binary:
				possible_value.append(int(character))
			possible_values.append(possible_value)
	
	if condition[0] == 3: #exact value function
		N = condition[1][0] #how many variables, each can have 0 or 1
		num_combinations = int(math.pow(2, N)) #How many possible combinations can exist

		#Find the binary representation of each possibility, which can be used as values for each variable
		for x in range(0,num_combinations):
			binary = ('000000000000000000000000000000' + str(bin(x))[2:])[-N:]
			possible_value = []
			for character in binary:
				possible_value.append(int(character))
			possible_values.append(possible_value)

	if condition[0] == 4: #Counting function
		N = condition[1][0] #how many variables, each can have 0 or 1
		num_combinations = int(math.pow(2, N)) #How many possible combinations can exist

		#Find the binary representation of each possibility, which can be used as values for each variable
		for x in range(0,num_combinations):
			binary = ('000000000000000000000000000000' + str(bin(x))[2:])[-N:]
			possible_value = []
			for character in binary:
				possible_value.append(int(character))
			possible_values.append(possible_value)

	if condition[0] == 5: #Nested
		possible_values = [[0,0], [0,1], [1,1]]

	return possible_values
	

#Returns true if a condition is true, false otherwise
def isTrue(instance, conditions):
	#Names of the features
	variable_names = string.ascii_uppercase

	#Used to keep track of which variable is currently examined
	current_var_id = 0
	
	for condition in conditions:
		#Determine what the condition is

		#Do this
		if condition[0] == 0: #Random
			#Do nothing
			current_var_id += 1

		if condition[0] == 1: #Parity function
			m = condition[1][0]
			counter = 0
			for x in range(0,m):
				counter += instance[variable_names[current_var_id+x]]
			if counter % 2 == 0:
				return False

		if condition[0] == 2: #M-out-of-N
			m = condition[1][1]
			n = condition[1][0]
			counter = 0
			for x in range(0, n):
				counter += instance[variable_names[current_var_id+x]]
			if counter < m:
				return False
			current_var_id += n #N
		
		if condition[0] == 3: #Exact value
			m = condition[1][1]
			n = condition[1][0]
			counter = 0
			for x in range(0,n):
				counter += instance[variable_names[current_var_id+x]]
			if counter != m:
				return False

		if condition[0] == 4: #Counting
			k = condition[1][1]
			n = condition[1][0]
			m = condition[1][2]
			counter = 0
			for x in range(0,n):
				counter += instance[variable_names[current_var_id+x]]
			if counter%m != k:
				return False

		if condition[0] == 5: #Nested implication
			if instance[variable_names[current_var_id+1]] == 0:
				return False
			current_var_id += 2

	return True

#Returns true if a condition is true, false otherwise
def getOutcome(values, condition):
	#Names of the features
	variable_names = string.ascii_uppercase
	
	#Determine what the condition is
	if condition[0] == 1: #Parity function
		m = condition[1][0]
		counter = 0
		for x in range(0,m):
			counter += values[x]
		if counter % 2 == 0:
			return False

	if condition[0] == 2: #M-out-of-N
		m = condition[1][1]
		n = condition[1][0]
		counter = 0
		for x in range(0, n):
			counter += values[x]
		if counter < m:
			return False
	
	if condition[0] == 3: #Exact value
		m = condition[1][1]
		n = condition[1][0]
		counter = 0
		for x in range(0,n):
			counter += values[x]
		if counter != m:
			return False

	if condition[0] == 4: #Counting
		k = condition[1][1]
		n = condition[1][0]
		m = condition[1][2]
		counter = 0
		for x in range(0,n):
			counter += [x]
		if counter%m != k:
			return False

	if condition[0] == 5: #Nested implication
		if values[1] == 0:
			return False

	return True


#Creates an instance using a set of conditions
def createInstance(instance_id, output, num_noise_vars, conditions, possible_values):
	instance = {}
	variable_names = string.ascii_uppercase
	current_var_id = 0

	for condition_id, condition in enumerate(conditions):
		if condition[0] == 0:
			instance, current_var_id = RandomCondition(instance, output, condition[1], current_var_id, (instance_id%len(conditions)==0), possible_values[condition_id], instance_id)
		if condition[0] == 1:
			instance, current_var_id = ParityFunction(instance, output, condition[1], current_var_id, (instance_id%len(conditions)==0), possible_values[condition_id], instance_id)
		if condition[0] == 2:
			instance, current_var_id = MOutOfNCondition(instance, output, condition[1], current_var_id, (instance_id%len(conditions)==0), possible_values[condition_id], instance_id)
		if condition[0] == 3:
			instance, current_var_id = ExactValueFunction(instance, output, condition[1], current_var_id, (instance_id%len(conditions)==0), possible_values[condition_id], instance_id)
		if condition[0] == 4:
			instance, current_var_id = CountingFunction(instance, output, condition[1], current_var_id, (instance_id%len(conditions)==0), possible_values[condition_id], instance_id)
		if condition[0] == 5:
			instance, current_var_id = NestedCondition(instance, output, condition[1], current_var_id, (instance_id%len(conditions)==0), possible_values[condition_id], instance_id)

	#Add the noise variables (Change this to noise1, noise2 etc)
	for x in range(0, num_noise_vars):
		variable_name = variable_names[current_var_id]
		current_var_id += 1
		instance[variable_name] = random.randint(0, 100)/100

	instance['Z'] = output

	return instance

def createInstanceEasy(output, possible_values, conditions):
	names = string.ascii_uppercase
	all_possible_values = possible_values[-1]
	if output == 1:
		values = all_possible_values[1][random.randint(0, len(all_possible_values[1])-1)]
	else:
		values = all_possible_values[2][random.randint(0, len(all_possible_values[2])-1)]

	instance = {}
	for val_id, val in enumerate(values):
		instance[names[val_id]] = val

	if isTrue(instance, conditions):
		instance['Z'] = 1
	else:
		instance['Z'] = 0
	return instance

def createDB(num_instances, output_ratio, num_noise_vars, conditions, possible_values):
	db = []
	output = 1
	for instance_id in range(0, num_instances):
		if instance_id > num_instances * output_ratio:
			output = 0
		instance = createInstanceEasy(output, possible_values, conditions)
		db.append(instance)
	return db

def createTestDB(num_instances, conditions, condition_id, features, num_noise_vars, all_possible_values):
	db = []

	#Names of the features
	variable_names = string.ascii_uppercase

	#Used to determine what the value of the varied variables will get
	variation_counter = 0

	num_instances = len(all_possible_values[condition_id][0])

	for instance_id in range(0, num_instances):
		#The new instance
		instance = {}

		#Used to keep track of which variable is currently examined
		current_var_id = 0

		for x, condition in enumerate(conditions):
			if x == condition_id: #These variables need to be varied

				#Find all the possible values that the variable can have
				possible_values = all_possible_values[x][0]
				current_values = possible_values[variation_counter]

				#Give the variables the right values
				for var_id in range(0, len(current_values)):
					instance[features[current_var_id]] = current_values[var_id]
					current_var_id += 1

				variation_counter += 1
				if variation_counter > len(possible_values)-1:
					variation_counter = 0

			else: #These variables need to satisfy their conditions
				if condition[0] == 0: #Random
					instance[features[current_var_id]] = random.randint(0, 100)
					current_var_id += 1

				if condition[0] == 1: #Parity
					possible_values = all_possible_values[x]
					values = possible_values[1][random.randint(0,len(possible_values[1])-1)]
					for val_id, val in enumerate(values):
						instance[variable_names[current_var_id+val_id]] = val
					current_var_id += condition[1][0]

				if condition[0] == 2: #M-out-of-N
					possible_values = all_possible_values[x]
					values = possible_values[1][random.randint(0,len(possible_values[1])-1)]
					for val_id, val in enumerate(values):
						instance[variable_names[current_var_id+val_id]] = val
					current_var_id += condition[1][0]
				
				if condition[0] == 3: #Exact value
					possible_values = all_possible_values[x]
					values = possible_values[1][random.randint(0,len(possible_values[1])-1)]
					for val_id, val in enumerate(values):
						instance[variable_names[current_var_id+val_id]] = val
					current_var_id += condition[1][0]
				
				if condition[0] == 4: #Counting
					possible_values = all_possible_values[x]
					values = possible_values[1][random.randint(0,len(possible_values[1])-1)]
					for val_id, val in enumerate(values):
						instance[variable_names[current_var_id+val_id]] = val
					current_var_id += condition[1][0]

				if condition[0] == 5: #Nested implication
					instance[features[current_var_id]] = random.randint(0, 1)
					instance[features[current_var_id+1]] = 1
					current_var_id += 2
		
		#Add the noise variables (Change this to noise1, noise2 etc)
		for x in range(0, num_noise_vars):
			variable_name = variable_names[current_var_id]
			current_var_id += 1
			instance[variable_name] = random.randint(0, 100)/100

		instance['Z'] = 1 if isTrue(instance, conditions) else 0
		db.append(instance)

	return db


def addBooleanNoise(db, conditions, noise_level = 0.1):
	noise_counter = 0
	new_db = []
	output_name = 'Z'
	for instance_id, instance in enumerate(db):
		if random.uniform(0, 1) <= noise_level:
			#add noise (invert the value of one feature, but not the output value)
			noise_counter += 1
			noisy_feature = random.randint(0,len(instance)-2)
			if list(instance.keys())[noisy_feature] == output_name: noisy_feature += 1
			noisy_feature = list(instance.keys())[noisy_feature]
			instance[noisy_feature] = 1 - instance[noisy_feature]
	print(repr(noise_counter) + ' instances with noise: ' + repr(100*noise_counter/len(db)) + ' percent')	
	findMistakes(db, conditions)
	return db

#Finds how many instances have an incorrect output label after adding noise
def findMistakes(db, conditions):
	output_name = 'Z'
	cor_counter = 0
	incor_counter = 0
	for instance in db:
		if (isTrue(instance, conditions) and instance[output_name] == 1) or  (not isTrue(instance, conditions) and instance[output_name] == 0):
			cor_counter += 1
		else:
			incor_counter += 1
	print('out of ' + repr(len(db)) + ' instance, ' + repr(cor_counter) + ' instances are correct and '+ repr(incor_counter) + ' instances are incorrect, which is ' + repr(100*incor_counter/len(db)) + ' percent incorrect')

def generateAllPossibleValues(conditions):
	num_variables = 0
	for condition in conditions:
		num_variables += condition[1][0]
	num_combinations = int(math.pow(2, num_variables))
	all_vals = []
	true_vals = []
	false_vals = []
	for x in range(0,num_combinations):
		binary = ('000000000000000000000000000000' + str(bin(x))[2:])[-num_variables:]
		possible_value = []
		for character in binary:
			possible_value.append(int(character))
		possible_instance = intToDict(possible_value)
		if isTrue(possible_instance, conditions):
			true_vals.append(possible_value)
		else:
			false_vals.append(possible_value)
		all_vals.append(possible_value)
	return [all_vals, true_vals, false_vals]

def intToDict(number):
	instance = {}
	variable_names = string.ascii_uppercase
	for idx, x in enumerate(number):
		instance[variable_names[idx]] = x
	return instance


def generateDatasets(params):

	print('\n####\nGenerating DB\nNumber of training instances: ' + repr(params['num_train_instances']) + '\nNumber of testing instances: ' + repr(params['num_test_instances']) +  '\nOutput ratio: ' + repr(params['output_ratio']) + '\nNumber of noise attributes: ' + repr(params['num_noise_vars']) + '\n')


	conditions = params['conditions']
	experiment_name = params['experiment_name']
	printConditions(conditions)

	#Generate all of the possible values for each condition to reduce computation time
	possible_values = []
	for condition in conditions:
		vals = generatePossibleValues(condition)
		true_vals = []
		false_vals = []
		for val in vals:
			if getOutcome(val, condition):
				true_vals.append(val)
			else:
				false_vals.append(val)
		possible_values.append([vals, true_vals, false_vals])

	#Add all of the possible values
	possible_values.append(generateAllPossibleValues(conditions))

	#Make folders if they don't exist already
	if not os.path.exists('../datasets/' + experiment_name):
		os.makedirs('../datasets/' + experiment_name)

	exportConditions(conditions, '../datasets/' + experiment_name + '/conditions.out')
	conditions = importConditions('../datasets/' + experiment_name + '/conditions.out')

	#Create a training dataset using the conditions and variables
	train_db = createDB(params['num_train_instances'], params['output_ratio'], params['num_noise_vars'], conditions, possible_values)
	if params['noise_level']: train_db = addBooleanNoise(train_db, conditions, params['noise_level'])

	print('# Created training set: ' + repr(len(train_db)) + ' instances')
	if params['showConditionVars']:
		exportCSV(changeFeatureNames(train_db, conditions, params['num_noise_vars']), '../datasets/' + experiment_name + '/train_db.csv')
	else:
		exportCSV(train_db, '../datasets/' + experiment_name + '/train_db.csv')

	#Create a general testing dataset using the same conditions
	general_test_db = createDB(params['num_test_instances'], params['output_ratio'], params['num_noise_vars'], conditions, possible_values)
	if params['noise_level']: general_test_db = addBooleanNoise(general_test_db, conditions, params['noise_level'])
	
	print('# Created general test set: ' + repr(len(general_test_db)) + ' instances')
	if params['showConditionVars']:
		exportCSV(changeFeatureNames(general_test_db, conditions, params['num_noise_vars']), '../datasets/' + experiment_name + '/test_db.csv')
	else:
		exportCSV(general_test_db, '../datasets/' + experiment_name + '/test_db.csv')

	#Create test sets for each condition
	features = list(train_db[0].keys())
	for condition_id, condition in enumerate(conditions):
		test_db = createTestDB(params['num_test_instances'], conditions, condition_id, features, params['num_noise_vars'], possible_values)
		print('# Created ' + repr(ConditionToName(condition)) +' test set: ' + repr(len(test_db)) + ' instances')
		if params['showConditionVars']:
			exportCSV(changeFeatureNames(test_db, conditions, params['num_noise_vars']), '../datasets/' + experiment_name + '/test_db_' + repr(condition_id) + '.csv')
		else:
			exportCSV(test_db, '../datasets/' + experiment_name + '/test_db_' + repr(condition_id) + '.csv')

	print('\nCreated all datasets\n####\n')


if __name__ == "__main__":

	#Variables used:
	################

	#Number of instances of the train set
	num_train_instances = 3000

	#Number of instances of the test set
	num_test_instances = 2000

	#Ratio of output variable (true to false)
	output_ratio = 0.5

	#Number of noise variables
	num_noise_vars = 0#3

	#conditions are lists of conditions, with in turn are lists that consists of a condition id (int) and a list of condition parameters
	conditions = [[3,[1,3]]]

	#The name of the experiment
	experiment_name = 'Experiment_1'

	#Show the conditions in the variable names?
	showConditionVars = True

	if len(sys.argv) < 3:
		num_train_instances = int(sys.argv[1])
		print('! Please provide db specifications as follows:\n1. Number of instances of the training set\n2. Number of instances of the training set\n3. Output ratio \n4. Number of noise attributes \n5. The conditions \n6. The name of the experiment')
	else:
		num_train_instances = int(sys.argv[1])
		num_test_instances = int(sys.argv[2])
		output_ratio = float(sys.argv[3])
		num_noise_vars = int(sys.argv[4])
		conditions = sys.argv[5]
		experiment_name = sys.argv[6]

	generateDatasets(num_train_instances, num_test_instances, output_ratio, num_noise_vars, conditions, experiment_name, showConditionVars)


