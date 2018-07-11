import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import math
import copy
from random import shuffle
import sys
import pickle

class RuleItem:
	def __init__(self, A, B, db):
		self.A = A
		self.B = B

	def calcSupportA(self, db):
		counter = 0
		for instance in db:
			if containsItems(instance, self.A):
				counter += 1
		support = float(counter)/float(len(db))
		return support

	def calcSupportB(self, db):
		counter = 0
		for instance in db:
			if containsItems(instance, [self.B]):
				counter += 1
		support = float(counter)/float(len(db))
		return support

	def calcSupportAB(self, db):
		counter = 0
		for instance in db:
			if containsItems(instance, self.A) and containsItems(instance, [self.B]):
				counter += 1
		support = float(counter)/float(len(db))
		return support

	def calcConfidence(self, db):
		try:
			return self.calcSupportAB(db)/self.calcSupportA(db)
		except ZeroDivisionError:
			return 0

	def calcLift(self, db):
		try:
			return self.calcSupportAB(db)/(self.calcSupportA(db)*self.calcSupportB(db))
		except ZeroDivisionError:
			return 0

	def calcConviction(self, db):
		try:
			return (1-self.calcSupportB(db))/(1-self.calcConfidence(db))
		except ZeroDivisionError:
			return 0

	def print(self, db=[], extra=False):
		if db == []:
			print(repr(self.A) + ' -> ' + repr(self.B))
		else:
			if extra == True:
				print(repr(self.A) + ' -> ' + repr(self.B) + ' (c=' + repr(round(self.calcConfidence(db),2)) + ", s=" + repr(round(self.calcSupportAB(db),2)) + ", l=" + repr(round(self.calcLift(db),2)) + ", v=" + repr(round(self.calcConviction(db),2))  +")")
			else:
				print(repr(self.A) + ' -> ' + repr(self.B) + ' (c=' + repr(round(self.calcConfidence(db),2)) + ", s=" + repr(round(self.calcSupportAB(db),2)) +")")

	def getRepr(self, db=[], extra=False):
		if db == []:
			return(repr(self.A) + ' -> ' + repr(self.B))
		else:
			if extra == True:
				return(repr(self.A) + ' -> ' + repr(self.B) + ' (c=' + repr(round(self.calcConfidence(db),2)) + ", s=" + repr(round(self.calcSupportAB(db),2)) + ", l=" + repr(round(self.calcLift(db),2)) + ", v=" + repr(round(self.calcConviction(db),2))  +")")
			else:
				return(repr(self.A) + ' -> ' + repr(self.B) + ' (c=' + repr(round(self.calcConfidence(db),2)) + ", s=" + repr(round(self.calcSupportAB(db),2)) +")")


#Reads a database as a list of dictionaries
def readCSV(path):
	with open(path) as input_file:
		reader = csv.DictReader(input_file)
		db = [row for row in reader]
	return db 

#Exports the database to a csv file
def exportCSV(database, path):
	keys = database[0].keys()
	with open(path, 'w', newline='') as output:
		dict_writer = csv.DictWriter(output, keys, delimiter=',')
		dict_writer.writeheader()
		dict_writer.writerows(database)

def exportOutputs(outputs, filename, features):
	features.append('output')
	outputs = pd.DataFrame(outputs)
	outputs.columns = features
	outputs.to_csv(filename)

def exportRules(rules, path):
	with open(path, 'wb') as fp:
		pickle.dump(rules, fp)

def importRules(path):
	with open (path, 'rb') as fp:
		return pickle.load(fp)

def exportClassifier(classifier, path, dfpath):
	rules = classifier[0]
	default_class = classifier[1]
	with open(path, 'wb') as fp:
		pickle.dump(rules, fp)
	with open(dfpath, 'wb') as fp:
		pickle.dump(default_class, fp)
	open(path[:-3]+'txt', 'w').close()
	f = open(path[:-3]+'txt','w')
	for r in rules:
		f.write(r.getRepr()+'\n')
	f.close()


def importClassifier(path, dfpath):
	rules = []
	default_class = ''
	with open (path, 'rb') as fp:
		rules = pickle.load(fp)
	with open (dfpath, 'rb') as fp:
		default_class = pickle.load(fp)
	return [rules, default_class]

def getOutputValues(db, output_name):
	unique_vals = []
	for instance in db:
		for feature in instance:
			if feature == output_name and instance[feature] not in unique_vals:
				unique_vals.append(instance[feature])
	return unique_vals

def getOutputValue(instance, output_name):
	for feature, value in instance:
		if feature == output_name:
			return value
	return False

def toNumpyArray(db):
	df = pd.DataFrame(db)
	x = df.as_matrix()
	return np.asarray(x)

#Each instance is converted into a list containing a list of items and the output value
def toItems(db, output_name):
	print('- Converting to Items... ')
	item_db = []
	unique_items = []

	for idx, instance in enumerate(db):
		sys.stdout.write('\r  '+repr(round(100*(idx+1)/len(db), 2)) + '%')
		items = []
		for feature in instance:
			item = [feature, instance[feature]]
			items.append(np.asarray([feature, instance[feature]]))
			if not containsItems(unique_items, [item]):
				unique_items.append(item)
		item_db.append(items)

	print('\n+ Conversion completed')
	item_db = np.asarray(item_db)
	return item_db, unique_items

#returns true if a set of items is in an instance
def containsItems(instance, item_set):
	counter = 0
	for item in item_set:
		for i in instance:
			if i[0] == item[0] and i[1] == item[1]:
				counter +=1
	if counter == len(item_set):
		return True
	return False

#Removes an item from an itemset (and all of the items with the same feature)
def removeItem(item_set, item):
	new_set = item_set
	for i in new_set:
		if i[0] == item[0]:
			new_set.remove(i)
	return new_set

#Sort a list of rules based on their confidence. If the confidence of two rules is the same, sort on support
def sortOnPrecedence(rules, db):
	print('- Sorting...')
	sorted_list = []
	for idr, r in enumerate(rules):
		sorted_list.append((r, r.calcConfidence(db), r.calcSupportAB(db)))

	sorted_list.sort(key=lambda row: row[2], reverse=True)
	sorted_list.sort(key=lambda row: row[1], reverse=True)

	sorted_rules = []
	for item in sorted_list:
		sorted_rules.append(item[0])
	print('+ Sorting completed')
	return sorted_rules


#Sort a list of rule structures based on their confidence. If the confidence of two rules is the same, sort on support
def sortStructOnPrecedence(rules, db):
	print('- Sorting...')
	sorted_list = []
	for idr, r in enumerate(rules):
		sorted_list.append((r, r[0].calcConfidence(db), r[0].calcSupportAB(db)))

	sorted_list.sort(key=lambda row: row[2], reverse=True)
	sorted_list.sort(key=lambda row: row[1], reverse=True)

	sorted_rules = []
	for item in sorted_list:
		sorted_rules.append(item[0])
	print('+ Sorting completed')
	return sorted_rules

def prune(rules, db, output_name):
	new_rules = []
	for r in rules:
		new_r = pessimisticErrorPrune(r, db, output_name)
		if not new_r in rules:
			new_rules.append(new_r)
	return new_rules

#Prunes a rule based on the pessimistic error (I think)
def pessimisticErrorPrune(rule, db, output_name):
	r = copy.deepcopy(rule)
	previous_rule = RuleItem(r.A[0:len(r.A)-2], r.B, db)
	current_error = classifySet([[rule], False], db, output_name)[1]
	previous_error = classifySet([[previous_rule], False], db, output_name)[1]
	if current_error <= previous_error:
		return rule
	else:
		return pessimisticErrorPrune(previous_rule, db, output_name)

#The apriori algorithm that generates a list of classification association rules (CARs)
def apriori(db, unique_items, output_values, output_name, support_threshold, confidence_threshold):
	CAR = []
	next_features = []
	print('- Starting Apriori...')

	print('  Intial rules')
	#Generate the rule items with 1 condition
	for idx, output_val in enumerate(output_values):
		sys.stdout.write('\r  '+repr(round(100*(idx+1)/len(output_values), 2)) + '%')
		output_item = [output_name, str(output_val)]
		for item in unique_items:
			if item[0] == output_name:
				continue
			r = RuleItem([item], output_item, db)
			if r.calcSupportAB(db) >= support_threshold:
				CAR.append(r)
				next_features.append(removeItem(copy.deepcopy(unique_items), item))
	print('\n  '+ repr(len(CAR)) + ' intial rules generated')
	print('  Proceeding with Apriori')

	previous_CAR = copy.deepcopy(CAR)
	num_next_features = len(next_features)

	for x in range(num_next_features):
		sys.stdout.write('\r  '+repr(round(100*(x+1)/float(num_next_features), 2)) + '%')
		new_CAR = []
		new_next_features = []
		#Generate candidate rules
		for idr, r in enumerate(previous_CAR):
			next_feature = next_features[idr][0]
			if next_feature[0] == output_name:
				continue
			new_A = r.A
			new_A.append(next_feature)
			new_r = RuleItem(new_A, r.B, db)
			if new_r.calcSupportAB(db) >= support_threshold:
				CAR.append(new_r)
				new_CAR.append(new_r)
				new_next_features.append(removeItem(copy.deepcopy(next_features[idr]), next_features[idr][0]))
		new_CAR = prune(new_CAR, db, output_name)
		previous_CAR = copy.deepcopy(new_CAR)
		next_features = new_next_features

	print('\n+ Apriori Finished: ' + repr(len(CAR)) + ' rules generated')

	CAR = sortOnPrecedence(CAR ,db)

	return CAR


#Finds distribution of the classes within the db
def compClassDistri(db, output_name):
	distribution = {}
	for instance in db:
		label = getOutputValue(instance, output_name)
		if label in distribution:
			distribution[label] += 1
		else:
			distribution[label] = 1
	return distribution

#Finds the most common class of the db
def mostCommonClass(db, output_name):
	distribution = compClassDistri(db, output_name)
	if distribution:
		return max(distribution, key=distribution.get)
	else:
		return ''

#Classifies an instance using a rule and default class
def classify(rule, instance):
	num_conditions = len(rule.A)
	counter = 0

	for precedent in rule.A:
		for feature, value in instance:
			if precedent[0] == feature and precedent[1] == value:
				counter += 1
	if counter == num_conditions:
		return rule.B[1]
	else:
		return False#default_class

#Removes all occurences of a list of instances within the db
def removeInstances(db, instances):
	new_db = []
	for idx, new_instance in enumerate(db):
		for instance in instances:
			identical_features = 0
			for feature, value in instance:
				for new_feature, new_value in new_instance:
					if feature == new_feature and value == new_value:
						identical_features += 1
		if identical_features < len(instance)-1:
			new_db.append(new_instance)
	db = np.asarray(new_db)
	return db

#Builds a classifier from the rules. M2 version is faster
def buildClassifier(rules, db, output_name):
	print('- Building classifier...')

	#The classifier
	classifier = []

	#The error of each rule
	errors = []

	#Make a copy of the database to be sure
	db_copy = copy.deepcopy(db)

	#The most common class is the default class
	default_class = mostCommonClass(db_copy, output_name)
	df_list = []

	#For each rule
	for idr, r in enumerate(rules):

		#List to store instances that are classified correctly by the rule
		temp = []

		#Whether the rule is marked
		marked = False

		for idi, instance in enumerate(db_copy):
			current_round = (len(db_copy)*idr) + idi
			sys.stdout.write('\r  '+repr(round(100*(current_round+1)/(len(db_copy)*len(rules)), 2)) + '%')
			#Get the true label of that instance
			true_label = getOutputValue(instance, output_name)

			#Classify each instance using the rule
			label = classify(r, instance)

			#If it is correct, add the instance to temp and set marked to true.
			#Else increase the error by 1
			if label == true_label:
				temp.append(instance)
				marked = True

		#If the rule is marked, append it to the classifier, append its error and recalculate the default class
		if marked is True:
			classifier.append(r)
			default_class = mostCommonClass(db_copy, output_name)
			labels, error = classifySet([classifier, default_class], db, output_name)
			errors.append(error)
			df_list.append(default_class)
			db_copy = removeInstances(db_copy, temp)
		
		if len(db_copy) == 0:
			break
		else:
			default_class = mostCommonClass(db_copy, output_name)

	max_error = 0
	last_rule_id = 0
	for ide, error in enumerate(errors):
		if error < max_error:
			last_rule_id = ide
		else:
			max_error = error

	classifier = classifier[:last_rule_id+1]
	default_class = df_list[last_rule_id]
	print('\n+ Classifier created with ' + repr(len(classifier)) + ' rules')

	return [classifier, default_class]

#Classifies a set of instances using the classifier, return both the labels and the error
def classifySet(classifier, test_set, output_name, output=False, original_db=False):
	#Extract the rules and the default class from the classifier
	rules = classifier[0]
	default_class = classifier[1]

	#Used to store the output labels
	all_labels = []

	#The number of correct classifications
	correct = 0

	if output == True:
		print('- Starting classification...')

	#For each instance in the test set
	for idx, instance in enumerate(test_set):
		if output == True:	
			sys.stdout.write('\r  '+repr(round(100*(idx+1)/len(test_set), 2)) + '%')

		#The label is the default class by default
		label = default_class

		#get the true label of the instance
		true_label = getOutputValue(instance, output_name)

		#For ech rule in the classifier
		for r in rules:
			satisfies = True
			precedent = r.A

			#The rule is not applicable if the feature of the instance has a different value
			for feature, value in precedent:
				for feature2, value2 in instance:
					if feature == feature2 and value != value2:
						#print(feature, value, feature2, value2)
						satisfies = False

			#If the rule is applicable, use it to classify the instance
			if satisfies is True:
				label = r.B[1]
				break
		#Add the label to the 
		all_labels.append(label)

		#Check if it was correct
		if label == true_label:
			correct += 1

	#Print the number of correct 
	if output == True:
		print('\n+ Classification completed')
		print('  Classification accuracy: ' + repr(round(100*correct/len(test_set), 2)) + '%')

	all_labels = np.asarray([all_labels]).T
	classified_instances = all_labels
	
	if original_db != False:
		new_db = toNumpyArray(original_db)
		classified_instances = np.concatenate((new_db, all_labels), axis=1)

	return [classified_instances, len(test_set) - correct]


def maxCoverRules(d, rules, true_label):
	cRule = False
	wRule = False

	for rid, r in enumerate(rules):
		label = classify(r, d)
		if label == true_label:
			cRule = rid
			break
	for rid, r in enumerate(rules):
		label = classify(r, d)
		if label == False:
			continue
		if label != true_label:
			wRule = rid
			break

	return cRule, wRule

def getTrueLabel(instance, output_name):
	true_label = False
	for feature, value in instance:
		if feature == output_name:
			true_label = value
	return true_label

def precedes(ruleA, ruleB):
	if ruleA.calcConfidence(db) > ruleB.calcConfidence(db):
		return True
	if ruleA.calcConfidence(db) < ruleB.calcConfidence(db):
		return False
	if ruleA.calcSupportAB(db) > ruleB.calcSupportAB(db):
		return True
	return False

#Finds all the rules in U that wrongly classify the instance and have higher precedence than cRule.
def allCoverRules(U, instance, cRule_struct, true_label):
	wSet = []
	for [rule, mark, classCasesCovered, replace, rid] in U:
		label = classify(rule, instance)
		if label == False:
			continue
		if label != true_label:
			wSet.append(rid)
			break
	return wSet

def printStruct(struct):
	print('* Rule '+ repr(struct[4]))
	struct[0].print(db=train_set, extra=False)
	print(' + ' + repr(struct[1]) + ', '+ repr(struct[2]) + ', ')
	for r in struct[3]:
		struct[0].print(db=train_set, extra=False)

def bestDefaultClass(db, classifier, output_name):
	remainder = []
	for instance in db:
		true_label = getTrueLabel(instance, output_name)
		counter = 0
		for rule in classifier:
			label = classify(rule, instance)
			if label == true_label:
				counter += 1
		if counter == 0:
			remainder.append(instance)
	default_class = mostCommonClass(remainder, output_name)
	return default_class

#Builds a classifier from the rules. M2 version is faster
def buildClassifierM2(rules, db, output_name):
	print('- Building classifier...')

	#The classifier
	classifier = []

	#The error of each rule
	errors = []

	#Make a copy of the database to be sure
	db_copy = copy.deepcopy(db)

	#The most common class is the default class
	default_class = mostCommonClass(db_copy, output_name)

	Q = []
	U = []
	A = []

	rulesStruct = []

	for rid, r in enumerate(rules):
		#1-rule, 2-Mark, 3-classCasesCovered, 4-replace, 5- id of the rule
		rulesStruct.append([r, False, {}, [], rid])
	
	#STAGE 1
	#print('STAGE ONE')
	for dID, d in enumerate(db_copy):
		true_label = getTrueLabel(d, output_name)

		#Find max cover rules
		cRule_id, wRule_id = maxCoverRules(d, rules, true_label)

		#Join cRule with U
		if rulesStruct[cRule_id] not in U:
			U.append(rulesStruct[cRule_id])

		#Increase the classCasesCovered
		if true_label in rulesStruct[cRule_id][2]:
			rulesStruct[cRule_id][2][true_label] += 1
		else:
			rulesStruct[cRule_id][2][true_label] = 1

		#If cRule > wRul (their ids are sorted based on precedence)
		if cRule_id < wRule_id or wRule_id == False:
			#Join cRule with Q
			if rulesStruct[cRule_id] not in Q:
				Q.append(rulesStruct[cRule_id])
			#Mark the cRule
			rulesStruct[cRule_id][1] = True
		else:
			#Join data structure with A
			if [dID, true_label, cRule_id, wRule_id] not in A:
				A.append([dID, true_label, cRule_id, wRule_id])

	if False:
		print('Q')
		for r in Q:
			printStruct(r)

		print('U')
		for r in U:
			printStruct(r)



	#STAGE 2
	#For each structure in A
	for dID, y, cRule_id, wRule_id in A:
		true_label = getTrueLabel(db_copy[dID], output_name)
		
		#If wRule is marked
		if rulesStruct[wRule_id][1] == True:
			#In/decrease the classCasesCovered
			if y in rulesStruct[cRule_id][2]:
				rulesStruct[cRule_id][2][y] -= 1
			if y in rulesStruct[wRule_id][2]:
				rulesStruct[wRule_id][2][y] += 1
			else:
				rulesStruct[wRule_id][2][y] = 1

		else:
			#Get the wSet
			wSet = allCoverRules(U, db_copy[dID], rulesStruct[cRule_id], true_label)
			for rid in wSet:
				#Add the cRule to the rule in the wSet
				rulesStruct[rid][3].append([rulesStruct[cRule_id], dID, y])

				#Increase classCasesCovered
				if true_label in rulesStruct[rid][2]:
					rulesStruct[rid][2][true_label] += 1
				else:
					rulesStruct[rid][2][true_label] = 1

				#Join wSet with U
				if rulesStruct[rid] not in Q:
					Q.append(rulesStruct[rid])

	#STAGE 3
	#print('STAGE THREE')
	Q = sortStructOnPrecedence(Q, db_copy)
	
	classDistr = compClassDistri(db_copy, output_name)
	ruleErrors = 0

	covered_rules = []

	default_class = mostCommonClass(db_copy, output_name)

	C = []
	C_rules_only = []

	for [rule, marked, classCasesCovered, replace, rid] in Q:
		if classCasesCovered[rule.B[1]] > 0:
			for [cRule, dID, y] in replace:
				if dID in covered_rules:
					rulesStruct[rid][2][y] -= 1
				else:
					cRule[2][y] -= 1
			C_rules_only.append(rule)
			ruleErrors = 0
			defaultErrors = 0
			default_class = bestDefaultClass(db_copy, C_rules_only, output_name)	
			totalErrors = ruleErrors + defaultErrors
			C.append([rule, default_class, totalErrors])

	previous_error = 10000000
	for idx, [rule, df, totalErrors] in enumerate(C):
		if totalErrors > previous_error:
			break
		else:
			classifier.append(rule)
			default_class = df
			previous_error = totalErrors

	print('\n+ Classifier created with ' + repr(len(classifier)) + ' rules')

	return [classifier, default_class]


if __name__ == "__main__":

	#Read in the database and divide it into a test and train set
	db = readCSV("../datasets/discrete_welfare.csv")
	db_onefail = readCSV("../datasets/discrete_welfare_onefail.csv")
	db_onefailtest = readCSV("../datasets/discrete_welfareTest_onefail.csv")
	db_test = readCSV("../datasets/discrete_welfareTest.csv")
	db_age = readCSV("../datasets/discrete_ageExperiment.csv")
	db_dist = readCSV("../datasets/discrete_distExperiment.csv")

	output_name = 'satisfied'

	train_set = db
	test_set = db_test

	shuffle(test_set)
	shuffle(train_set)

	#Increase confidence, decrease support
	min_support = 0.02
	min_confidence = 0.5

	output_values = getOutputValues(train_set, output_name)
	features = list(train_set[0].keys())
	test_set, unique_items = toItems(test_set, output_name)
	train_set, unique_items = toItems(train_set, output_name)
	
	#### Experiment
	#Training on training set a
	rules = apriori(train_set, unique_items, output_values, output_name, min_support, min_confidence)
	exportRules(rules, 'Rules/rulesA.cor')
	rules = importRules('Rules/rulesA.cor')
	classifier = buildClassifierM2(rules, train_set, output_name)
	exportClassifier(classifier, 'Classifiers/classifierA.cor', 'Classifiers/dcA.cor')
	classifier = importClassifier('Classifiers/classifierA.cor', 'Classifiers/dcA.cor')

	#Test set A
	outputs, error = classifySet(classifier, test_set, output_name, output=True, original_db=db_test)
	exportOutputs(outputs, 'Outputs/outputTestA.csv', list(db_test[0].keys()))

	#Test set B
	outputs, error = classifySet(classifier, toItems(db_onefailtest, output_name)[0], output_name, output=True, original_db=db_onefailtest)
	exportOutputs(outputs, 'Outputs/outputTest1FA.csv', list(db_onefailtest[0].keys()))

	#Age test set
	outputs, error = classifySet(classifier, toItems(db_age, output_name)[0], output_name, output=True, original_db=db_age)
	exportOutputs(outputs, 'Outputs/outputAgeA.csv', list(db_age[0].keys()))

	#Dist test set
	outputs, error = classifySet(classifier, toItems(db_dist, output_name)[0], output_name, output=True, original_db=db_dist)
	exportOutputs(outputs, 'Outputs/outputDistA.csv', list(db_dist[0].keys()))

	#Age test set
	outputs, error = classifySet(classifier, toItems(db_age_noise, output_name)[0], output_name, output=True, original_db=db_age_noise)
	exportOutputs(outputs, 'Outputs/outputAgeA.csv', list(db_age_noise[0].keys()))

	#Dist test set
	outputs, error = classifySet(classifier, toItems(db_dist_noise, output_name)[0], output_name, output=True, original_db=db_dist_noise)
	exportOutputs(outputs, 'Outputs/outputDistA.csv', list(db_dist_noise[0].keys()))

	###
	#Training on training set B
	train_set = db_onefail
	shuffle(train_set)
	output_values = getOutputValues(train_set, output_name)
	features = list(train_set[0].keys())
	train_set, unique_items = toItems(train_set, output_name)

	rules = apriori(train_set, unique_items, output_values, output_name, min_support, min_confidence)
	exportRules(rules, 'Rules/rulesB.cor')
	rules = importRules('Rules/rulesB.cor')
	classifier = buildClassifierM2(rules, train_set, output_name)
	exportClassifier(classifier, 'Classifiers/classifierB.cor', 'Classifiers/dcB.cor')
	classifier = importClassifier('Classifiers/classifierB.cor', 'Classifiers/dcB.cor')

	#Test set A
	outputs, error = classifySet(classifier, test_set, output_name, output=True, original_db=db_test)
	exportOutputs(outputs, 'Outputs/outputTestB.csv', features)

	#Test set B
	outputs, error = classifySet(classifier, toItems(db_onefailtest, output_name)[0], output_name, output=True, original_db=db_onefailtest)
	exportOutputs(outputs, 'Outputs/outputTest1FB.csv', list(db_onefailtest[0].keys()))

	#Age test set
	outputs, error = classifySet(classifier, toItems(db_age, output_name)[0], output_name, output=True, original_db=db_age)
	exportOutputs(outputs, 'Outputs/outputAgeB.csv', list(db_age[0].keys()))

	#Dist test set
	outputs, error = classifySet(classifier, toItems(db_dist, output_name)[0], output_name, output=True, original_db=db_dist)
	exportOutputs(outputs, 'Outputs/outputDistB.csv', list(db_dist[0].keys()))

