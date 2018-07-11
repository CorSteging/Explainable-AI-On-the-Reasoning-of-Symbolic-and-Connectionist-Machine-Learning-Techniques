import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import math
import numbers
from random import shuffle


class Question:

	def __init__(self, col, val, features):
		self.col = col
		self.val = val
		self.features = features

	def __repr__(self):
		if isNumber(self.val):
			return "Is " + repr(self.features[self.col]) + " greater or equal to " + repr(self.val) + "?"
		else:
			return "Is " + repr(self.features[self.col]) + " equal to " + repr(self.val) + "?"

	def isTrue(self, instance):
		x = instance[self.col]
		if isNumber(x):
			return float(x) >= float(self.val)
		else:
			return x == self.val

class Leaf:
	def __init__(self, db, output_col):
		self.predictions = getClassDistribution(db, output_col)

class Node: 
	def __init__(self, question, left_branch, right_branch):
		self.question = question
		self.left_branch = left_branch
		self.right_branch = right_branch


def getTXTReprentation(tree, file):
	if isinstance(tree, Leaf):
		file.write("OUTPUT" + repr(tree.predictions) +"\n")
		return
	file.write(repr(tree.question))
	file.write('\nIF TRUE ->\n')
	getTXTReprentation(tree.left_branch, file)
	file.write('ELSE ->\n')
	getTXTReprentation(tree.right_branch, file)

def exportTree(tree, filename):
	file = open("models/"+filename+".txt", "w")
	getTXTReprentation(tree, file)
	file.close()

def importTree(filename):
	return np.genfromtxt("models/test.txt")


def exportOutputs(outputs, filename, features):
	features.append('output')
	outputs = pd.DataFrame(outputs)
	outputs.columns = features
	outputs.to_csv(filename)


def isNumber(x):
	#return isinstance(x, numbers.Real)
	try:
		num = float(x)
		return True
	except ValueError:
		return False
	try:
		num = int(x)
		return True
	except ValueError:
		return False
	return True


def toNumpyArray(db):
	df = pd.DataFrame(db)
	x = df.as_matrix()
	return np.asarray(x)

#Find the column number of the output variable of a db
def findOutputCol(db, output_name):
	col = 0
	for x, key in enumerate(db[0]):
		if key == output_name:
			col = x
	return col

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

def GiniImpurity(db, output_col):
	distribution = getClassDistribution(db, output_col)
	imp = 1
	#For each class in the db
	for label in  distribution:
		imp -= (distribution[label]/float(len(db)))**2
	return imp

def infoGain(left, right, cur_gini, output_col):
	p = float(len(left)) / (len(left) + len(right))
	return cur_gini -p * GiniImpurity(left, output_col) - (1-p) * GiniImpurity(right, output_col)


def getClassDistribution(db, output_col):
	distribution = {}
	for instance in db:
		cur_class = instance[output_col]
		if cur_class in distribution:
			distribution[cur_class] += 1
		else:
			distribution[cur_class] = 0
	return distribution

def splitRows(db, question):
	row_true = []
	row_false = []

	for instance in db:
		if question.isTrue(instance):
			row_true.append(instance)
		else:
			row_false.append(instance)
	return row_true, row_false


def findBestQuestion(db, output_col, features):
	bestQuestion = None
	maximumGain = 0
	cur_gini = GiniImpurity(db, output_col)

	#For each features
	for feature, feature_name in enumerate(features):

		#Skip the output name
		if feature == output_col:
			continue

		#Find the unique values of that feature
		unique_values = set([instance[feature] for instance in db]) 
		
		#For each unique value
		for val in unique_values:

			#Make a question
			question = Question(feature, val, features)

			#Split the dataset
			left_set, right_set = splitRows(db, question)

			#If this doesn't split the dataset, continue
			if len(left_set) == len(db) or len(right_set) == len(db):
				continue

			#Calculate the gain
			gain = infoGain(left_set, right_set, cur_gini, output_col)

			#Decide whether it is good or bad
			if gain > maximumGain:
				maximumGain = gain
				bestQuestion = question

	return bestQuestion, maximumGain

def buildTree(db, output_col, features, gain_level=0, depth=0):
	question, gain = findBestQuestion(db, output_col, features)

	#If gain is 0, its a leaf
	if gain <= gain_level:# * pow(0.5, depth):
		return Leaf(db, output_col)

	#Split the db based on the question
	left_set, right_set = splitRows(db, question)

	left_branch = buildTree(left_set, output_col, features, gain_level, depth+1)
	right_branch = buildTree(right_set, output_col, features, gain_level, depth+1)

	return Node(question, left_branch, right_branch)


def REP(db, output_col, features):
	training = db[:round(len(db)*0.8)]
	validation = db[:round(len(db)*0.8)+1]
	tree = buildTree(training, output_col, features)

	return tree


def printTree(tree, features, blank=""):

	#The end of the tree branch
	if isinstance(tree, Leaf):
		printLeaf(tree, features, blank)
		return

	#Print question
	print(blank + repr(tree.question))

	#Print left
	print(blank + 'IF TRUE ->')
	printTree(tree.left_branch, features, blank + "   ")

	#Print right
	print(blank + 'ELSE ->')
	printTree(tree.right_branch, features, blank + "   ")


def printLeaf(leaf, features, blank=""):
	total = 0
	for key in leaf.predictions:
		total += leaf.predictions[key]
	if total > 0:
		for key in leaf.predictions:
			leaf.predictions[key] = round(leaf.predictions[key]/total*100,2)
	print(blank + "OUTPUT" + repr(leaf.predictions))


def classify(instance, tree):
	if isinstance(tree, Leaf):
		return tree.predictions
	if tree.question.isTrue(instance):
		return classify(instance, tree.left_branch)
	else:
		return classify(instance, tree.right_branch)

def classifySet(db, tree, output_col, show_output = False):
	correct = 0
	classified_instances = db
	labels = []
	for instance in db:
		output = classify(instance, tree)
		label = max(output, key=output.get)

		labels.append(label)

		if label == instance[output_col] or (label == '1' and instance[output_col] == '1.0') or (label == '0' and instance[output_col] == '0.0'): #hotfix sorry
			correct += 1

		if(show_output): print(repr(instance[output_col]) + " was given label " + repr(label))
	print("accuracy: " + repr(round(correct/len(db)*100,2)))

	labels = toNumpyArray(labels)
	classified_instances = np.concatenate((classified_instances, labels), axis=1)
	return classified_instances	



if __name__ == "__main__":

	noise_level = 5
	if False:
		db = readCSV("../datasets/"+ repr(noise_level*10)+ "pNoise/welfare_noise.csv")
		db_onefail = readCSV("../datasets/"+ repr(noise_level*10)+ "pNoise/welfare_onefail_noise.csv")
		db_onefailtest = readCSV("../datasets/"+ repr(noise_level*10)+ "pNoise/welfareTest_onefail_noise.csv")
		db_test = readCSV("../datasets/"+ repr(noise_level*10)+ "pNoise/welfareTest_noise.csv")
		db_age = readCSV("../datasets/"+ repr(noise_level*10)+ "pNoise/ageExperiment.csv")
		db_dist = readCSV("../datasets/"+ repr(noise_level*10)+ "pNoise/distExperiment.csv")
		db_age_noise = readCSV("../datasets/"+ repr(noise_level*10)+ "pNoise/ageExperiment_noise.csv")
		db_dist_noise = readCSV("../datasets/"+ repr(noise_level*10)+ "pNoise/distExperiment_noise.csv")


	#Read in the database and divide it into a test and train set
	db = readCSV("../datasets/welfare.csv")
	db_onefail = readCSV("../datasets/welfare_onefail.csv")
	db_onefailtest = readCSV("../datasets/welfareTest_onefail.csv")
	db_test = readCSV("../datasets/welfareTest.csv")
	db_age = readCSV("../datasets/ageExperiment.csv")
	db_dist = readCSV("../datasets/distExperiment.csv")

	output_name = 'satisfied'

	#### Trained on train set A
	features = list(db[0].keys())
	train_set = db
	shuffle(train_set)
	output_col = findOutputCol(train_set, output_name)
	train_set = toNumpyArray(train_set)
	tree = buildTree(train_set, output_col, features, gain_level = 0)
	printTree(tree, features)

	print('Test set A')
	outputs = classifySet(toNumpyArray(db_test), tree, output_col, show_output = False)
	exportOutputs(outputs, 'Outputs/outputsA.csv', list(db_test[0].keys()))

	print('Test set B')
	outputs = classifySet(toNumpyArray(db_onefailtest), tree, findOutputCol(db_onefailtest, output_name), show_output = False)
	exportOutputs(outputs, 'Outputs/outputs_failA.csv', list(db_onefailtest[0].keys()))

	print('Age set')
	outputs = classifySet(toNumpyArray(db_age), tree, findOutputCol(db_age, output_name), show_output = False)
	exportOutputs(outputs, 'Outputs/outputs_ageA.csv', list(db_age[0].keys()))

	print('Dist set')
	outputs = classifySet(toNumpyArray(db_dist), tree, findOutputCol(db_dist, output_name), show_output = False)
	exportOutputs(outputs, 'Outputs/outputs_distA.csv', list(db_dist[0].keys()))

	exportTree(tree, 'tree_A')

	#### Trained on train set B
	features = list(db_onefail[0].keys())
	train_set = db_onefail
	shuffle(train_set)
	output_col = findOutputCol(train_set, output_name)
	train_set = toNumpyArray(train_set)
	tree = buildTree(train_set, output_col, features, gain_level = 0)
	printTree(tree, features)

	print('Test set A')
	outputs = classifySet(toNumpyArray(db_test), tree, output_col, show_output = False)
	exportOutputs(outputs, 'Outputs/outputsB.csv', list(db_test[0].keys()))

	print('Test set B')
	outputs = classifySet(toNumpyArray(db_onefailtest), tree, findOutputCol(db_onefailtest, output_name), show_output = False)
	exportOutputs(outputs, 'Outputs/outputs_failB.csv', list(db_onefailtest[0].keys()))

	print('Age set')
	outputs = classifySet(toNumpyArray(db_age), tree, findOutputCol(db_age, output_name), show_output = False)
	exportOutputs(outputs, 'Outputs/outputs_ageB.csv', list(db_age[0].keys()))

	print('Dist set')
	outputs = classifySet(toNumpyArray(db_dist), tree, findOutputCol(db_dist, output_name), show_output = False)
	exportOutputs(outputs, 'Outputs/outputs_distB.csv', list(db_dist[0].keys()))

	exportTree(tree, 'tree_B')
