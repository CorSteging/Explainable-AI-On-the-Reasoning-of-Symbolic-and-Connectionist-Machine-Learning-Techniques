#Generates a fictional welfare benefit dataset based on the paper of Bench-Capon (1993)
#Author: Cor Steging

import csv
import random

import prepareData as prep


#Exports the database to a csv file
def exportCSV(database, filename):
	#Sort the keys of each 
	new_db = []
	keys = list(database[0].keys())
	print("\n", filename)
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


#Prints an instance nicely (without noise)
def getRepr(instance):
	nice_instance = {}
	features = list(instance.keys())
	for feature in features:
		if not 'attr' in feature:
			nice_instance[feature] = instance[feature]
	return nice_instance

#Returns true if a case is satisfied and says it is, or if it says it is not satisfied and is not
def testCase(case, satisfied):
	failed = False
	if case['age'] < 60 and case['gender'] == 'f':
		failed = True
	if case['age'] < 65 and case['gender'] == 'm':
		failed = True
	if case['paid_contributions'] < 4:
		failed = True
	if case['spouse'] < 1:
		failed = True
	if case['residence'] < 1:
		failed = True
	if case['capital'] > 2999:
		failed = True
	if case['distance'] >= 50 and case['patient_type'] == 'in':
		failed = True
	if case['distance'] < 50 and case['patient_type'] == 'out':
		failed = True

	if failed == True and satisfied == 1:
		#print("failed but said its true")
		return False
	if failed == False and satisfied == 0:
		#print("did not fail even though it should")
		return False
	return True

#TestCase is used for db's that don't undergo preparation (Variables are already seperated)
def testCase2(case, satisfied):
	failed = False
	if case['age'] < 60 and case['gender'] == 'f':
		failed = True
	if case['age'] < 65 and case['gender'] == 'm':
		failed = True
	if case['paid_contributions_4'] == 0 and case['paid_contributions_5'] == 0 :
		failed = True
	if case['spouse'] < 1:
		failed = True
	if case['residence'] < 1:
		failed = True
	if case['capital'] > 2999:
		failed = True
	if case['distance'] >= 50 and case['patient_type'] == 'in':
		failed = True
	if case['distance'] < 50 and case['patient_type'] == 'out':
		failed = True

	if failed == True and satisfied == 1:
		return False
	if failed == False and satisfied == 0:
		return False
	return True

#Generates a case (the fail feature determines what feature will fail for sure if the case is not satisfied)
def generateCase(satisfied, num_noise, fail_feature):
	fail_feature += 1
	case = {}

	#Satisfied
	case['satisfied'] = satisfied
	
	#Gender
	if random.randint(1,2) > 1:
		case['gender'] = 'm'
	else:
		case['gender'] = 'f'

	#Age
	if satisfied > 0:
		if case['gender'] == 'm':
			case['age'] = random.randint(65, 100)
		if case['gender'] == 'f':
			case['age'] = random.randint(60, 100)
	else:
		if fail_feature == 1:
			if case['gender'] == 'm':
				case['age'] = random.randint(0, 64)
			if case['gender'] == 'f':
				case['age'] = random.randint(0, 59)
		else:
			case['age'] = random.randint(0, 100)

	#Has the person paid contributions in 4 out of the last 5 contributions
	if satisfied > 0:
		case['paid_contributions'] = random.randint(4, 5)
	else:
		if fail_feature == 2:
			case['paid_contributions'] = random.randint(1, 3)
		else:
			case['paid_contributions'] = random.randint(1, 5)

	#Is the person a spouse of the patient
	if satisfied > 0:
		case['spouse'] = 1
	else:
		if fail_feature == 3:
			case['spouse'] = 0
		else:
			case['spouse'] = random.randint(0,1)

	#Does the person live in the UK
	if satisfied > 0:
		case['residence'] = 1
	else:
		if fail_feature == 4:
			case['residence'] = 0
		else:
			case['residence'] = random.randint(0,1)

	#Does the person have less than 3000 pounds in capital resources
	if satisfied > 0:
		case['capital'] = random.randint(0,2999)
	else:
		if fail_feature == 5:
			case['capital'] = random.randint(3000,10000)
		else:
			case['capital'] = random.randint(0,10000)

	#Is the relative an in patient or out patient
	if random.randint(1,2) > 1:
		case['patient_type'] = 0#'in'
	else:
		case['patient_type'] = 1#'out'

	#Distance to the hospital (must be less than 50 miles for in patient)
	if satisfied > 0:
		if case['patient_type'] == 0:#'in':
			case['distance'] = random.randint(0, 49)
		if case['patient_type'] == 1:#'out':
			case['distance'] = random.randint(50, 100)
	else:
		if fail_feature == 6:
			if case['patient_type'] == 0:#'in':
				case['distance'] = random.randint(50, 100)
			else:
				case['distance'] = random.randint(0, 49)
		else:
			case['distance'] = random.randint(0, 100)

	#Add noise attributes
	for i in range(1, 52):
		if(num_noise == 0):
			case['attr'+str(i)] = 0
		else:
			case['attr'+str(i)] = random.randint(1, 100)

	#RECURSION
	if testCase(case, satisfied) == False:
		case = generateCase(satisfied, num_noise, fail_feature)

	return 	case

#Generates the database
def generateDatabase(num_cases, proportion, num_noise):
	database = []
	for x in range(0, num_cases):
		satisfied = 1
		if x > num_cases*proportion: satisfied = 0
		database.append(generateCase(satisfied, num_noise, x%6))
	return database


########Used in the age experiment ########

#Generates a case for the age experiment
def generateAgeCase(num_noise, age, gender):
	case = {}

	#Satisfied
	case['satisfied'] = 0.0
	case['gender'] = gender
	case['age'] = age
	if gender =='m' and age >= 65:
		case['satisfied'] = 1.0
	if gender =='f' and age >= 60:
		case['satisfied'] = 1.0
	satisfied = case['satisfied']

	if random.randint(1,2) > 1:
		case['paid_contributions'] = 4.0
	else:
		case['paid_contributions'] = 5.0
	case['spouse'] = 1
	case['residence'] = 1
	case['capital'] = random.randint(0,2999)

	#Is the relative an in patient or out patient
	if random.randint(1,2) > 1:
		case['patient_type'] = "out"
	else:
		case['patient_type'] = "in"
	if case['patient_type'] == "in":
		case['distance'] = random.randint(0, 49)
	if case['patient_type'] == "out":
		case['distance'] = random.randint(50, 100)

	#Add noise attributes
	for i in range(1, 52):
		case['attr'+str(i)] = random.randint(1, 100)

	#RECURSION
	if testCase(case, satisfied) == False:
		case = generateAgeCase(num_noise, age, gender)

	return 	case

#Generates a prepared case for the age experiment
def generatePreparedAgeCase(num_noise, age, gender):
	case = {}

	#Satisfied
	case['satisfied'] = 0.0
	if gender =='m' and age > 65:
		case['satisfied'] = 1.0
	if gender =='f' and age > 60:
		case['satisfied'] = 1.0
	satisfied = case['satisfied']
	
	case['gender'] = 1.0
	if gender == 'f':
		case['gender'] = 0.0
	case['age'] = age/100
	if random.randint(1,2) > 1:
		case['paid_contributions_5'] = 1.0
		case['paid_contributions_4'] = 0.0
	else:
		case['paid_contributions_5'] = 1.0
		case['paid_contributions_4'] = 0.0		
	case['paid_contributions_3'] = 0.0
	case['paid_contributions_2'] = 0.0
	case['paid_contributions_1'] = 0.0
	case['spouse'] = 1
	case['spouse'] = random.randint(0,1)
	case['residence'] = 1
	case['capital'] = random.randint(0,2999)/10000

	#Is the relative an in patient or out patient
	if random.randint(1,2) > 1:
		case['patient_type'] = 0.0
	else:
		case['patient_type'] = 1.0
	if case['patient_type'] == 0.0:
		case['distance'] = random.randint(0, 49)/100
	if case['patient_type'] == 1.0:
		case['distance'] = random.randint(50, 100)/100

	#Add noise attributes
	for i in range(1, 52):
		case['attr'+str(i)] = random.randint(1, 100)/100

	#RECURSION
	if testCase2(case, satisfied) == False:
		case = generatePreparedAgeCase(num_noise, age, gender)

	return 	case

def generateAgeDatabase(num_cases, num_noise, age_step, prepared=True):
	database = []
	for i in range(0, int(num_cases/40)):
		age = age_step
		gender = 'm'
		for x in range(0, 40):
			age = age_step*(x+1)
			if x >= 20:
				gender = 'f'
				age -= 100
			if prepared is True:
				database.append(generatePreparedAgeCase(num_noise, age, gender))
			else:
				database.append(generateAgeCase(num_noise, age, gender))
	return database


########Used in the distance experiment ########

#Generates a case for the distance experiment
def generateDistCase(num_noise, dist, p_type):
	case = {}

	#Satisfied
	case['satisfied'] = 0.0
	if p_type =='in' and dist < 50:
		case['satisfied'] = 1.0
	if p_type =='out' and dist >= 50:
		case['satisfied'] = 1.0
	satisfied = case['satisfied']
	
	#1 = out, 0 = in
	case['patient_type'] = p_type
	case['distance'] = dist

	#Age/gender
	if random.randint(1,2) > 1:
		case['gender'] = 'm'
	else:
		case['gender'] = 'f'
	if case['gender'] == 'f':
		case['age'] = random.randint(60, 100)
	if case['gender'] == 'm':
		case['age'] = random.randint(65, 100)
	if random.randint(1,2) > 1:
		case['paid_contributions'] = 4.0
	else:
		case['paid_contributions'] = 5.0
	case['spouse'] = 1
	case['residence'] = 1
	case['capital'] = random.randint(0,2999)

	#Add noise attributes
	for i in range(1, 52):
		case['attr'+str(i)] = random.randint(1, 100)

	#RECURSION
	if testCase(case, satisfied) == False:
		case = generateDistCase(num_noise, dist, p_type)

	return 	case

#Generates a prepared case for the distance experiment
def generatePreparedDistCase(num_noise, dist, p_type):
	case = {}

	#Satisfied
	case['satisfied'] = 0.0
	if p_type =='in' and dist < 50:
		case['satisfied'] = 1.0
	if p_type =='out' and dist >= 50:
		case['satisfied'] = 1.0
	satisfied = case['satisfied']
	
	#1 = out, 0 = in
	case['patient_type'] = 1.0
	if p_type == 'in':
		case['patient_type'] = 0.0
	case['distance'] = dist/100

	#Age/gender
	if random.randint(1,2) > 1:
		case['gender'] = 0.0
	else:
		case['gender'] = 1.0
	if case['gender'] == 0.0:
		case['age'] = random.randint(60, 100)/100
	if case['gender'] == 1.0:
		case['age'] = random.randint(65, 100)/100

	if random.randint(1,2) > 1:
		case['paid_contributions_5'] = 1.0
		case['paid_contributions_4'] = 0.0
	else:
		case['paid_contributions_5'] = 1.0
		case['paid_contributions_4'] = 0.0		
	case['paid_contributions_3'] = 0.0
	case['paid_contributions_2'] = 0.0
	case['paid_contributions_1'] = 0.0
	case['spouse'] = 1
	case['spouse'] = random.randint(0,1)
	case['residence'] = 1
	case['capital'] = random.randint(0,2999)/10000


	#Add noise attributes
	for i in range(1, 52):
		case['attr'+str(i)] = random.randint(1, 100)/100

	#RECURSION
	if testCase2(case, satisfied) == False:
		case = generatePreparedDistCase(num_noise, dist, p_type)

	return 	case

def generateDistDatabase(num_cases, num_noise, dist_step, prepared=True):
	database = []
	for i in range(0, int(num_cases/40)):
		dist = dist_step
		p_type = 'in'
		for x in range(0, 40):
			dist = dist_step*(x+1)
			if x >= 20:
				p_type = 'out'
				dist -= 100
			if prepared is True:
				database.append(generatePreparedDistCase(num_noise, dist, p_type))
			else:
				database.append(generateDistCase(num_noise, dist, p_type))
	return database



########Used for cases with only 1 failed feature ########

#Generates a case where only one feature fails
def generateOneFailCase(satisfied, num_noise, fail_feature):
	case = {}
	fail_feature += 1

	#Satisfied
	case['satisfied'] = satisfied
	
	#Gender
	if random.randint(1,2) > 1:
		case['gender'] = 'm'
	else:
		case['gender'] = 'f'

	#Age
	if satisfied == 0  and fail_feature == 1:
		if case['gender'] == 'm':
			case['age'] = random.randint(0, 64)
		if case['gender'] == 'f':
			case['age'] = random.randint(0, 59)
	else:
		if case['gender'] == 'm':
			case['age'] = random.randint(65, 100)
		if case['gender'] == 'f':
			case['age'] = random.randint(60, 100)

	#Has the person paid contributions in 4 out of the last 5 contributions
	if satisfied == 0  and  fail_feature == 2:
		case['paid_contributions'] = random.randint(1, 3)
	else:
		case['paid_contributions'] = random.randint(4, 5)

	#Is the person a spouse of the patient
	if satisfied == 0  and fail_feature == 3:
		case['spouse'] = 0
	else:
		case['spouse'] = 1

	#Does the person live in the UK
	if satisfied == 0  and fail_feature == 4:
		case['residence'] = 0
	else:
		case['residence'] = 1

	#Does the person have less than 3000 pounds in capital resources
	if satisfied == 0  and fail_feature == 5:
		case['capital'] = random.randint(3000,10000)
	else:
		case['capital'] = random.randint(0,2999)

	#Is the relative an in patient or out patient
	if random.randint(1,2) > 1:
		case['patient_type'] = 'in'
	else:
		case['patient_type'] = 'out'

	#Distance to the hospital (must be less than 50 miles for in patient)
	if satisfied == 0  and fail_feature == 6:
		if case['patient_type'] == 'in':
			case['distance'] = random.randint(50, 100)
		else:
			case['distance'] = random.randint(0, 49)
	else:
		if case['patient_type'] == 'in':
			case['distance'] = random.randint(0, 49)
		if case['patient_type'] == 'out':
			case['distance'] = random.randint(50, 100)

	#Add noise attributes
	for i in range(1, 52):
		if(num_noise == 0):
			case['attr'+str(i)] = 0
		else:
			case['attr'+str(i)] = random.randint(1, 100)

	#RECURSION
	if testCase(case, satisfied) == False:
		case = generateOneFailCase(satisfied, num_noise, fail_feature)

	return 	case

#Generates the database
def generateOneFailDatabase(num_cases, proportion, num_noise):
	database = []
	for x in range(0, num_cases):
		satisfied = 1
		if x > num_cases*proportion: satisfied = 0
		database.append(generateOneFailCase(satisfied, num_noise, x%6))
	return database

#adds boolean noise to prepared or discrete data
def addNoise(db, noise_level = 0.0, discrete = False):
	noise_counter = 0
	new_db = []
	output_name = 'satisfied'

	features = list(db[0].keys())
	new_features = []
	for feature in features:
		if not 'attr' in feature and not output_name in feature:
			new_features.append(feature)
			
	for instance_id, instance in enumerate(db):
		if random.uniform(0, 1) <= noise_level:
			#add noise (invert the value of one feature, but not the output value)
			noise_counter += 1
			noisy_feature = new_features[random.randint(0,len(new_features)-1)]

			if instance[noisy_feature] == 'm':
				instance[noisy_feature] = 'f'
			elif instance[noisy_feature] == 'f':
				instance[noisy_feature] = 'm'
			if instance[noisy_feature] == 'in':
				instance[noisy_feature] = 'out'
			elif instance[noisy_feature] == 'out':
				instance[noisy_feature] = 'in'
			else:
				if instance[noisy_feature] <= 1.0:
					instance[noisy_feature] = 1 - instance[noisy_feature]
				else:
					instance[noisy_feature] = 100 - instance[noisy_feature]

	return db


#Adds boolean noise to the regular data sets (without preparation or discrete)
def addNoiseRegular(db, noise_level = 0.0, discrete = False):
	noise_counter = 0
	new_db = []
	output_name = 'satisfied'

	features = list(db[0].keys())
	new_features = []
	for feature in features:
		if not 'attr' in feature and not output_name in feature:
			new_features.append(feature)
			
	for instance_id, instance in enumerate(db):
		if random.uniform(0, 1) <= noise_level:
			#add noise (invert the value of one feature, but not the output value)
			noise_counter += 1
			noisy_feature = new_features[random.randint(0,len(new_features)-1)]
			if instance[noisy_feature] == 'm':
				continue
				instance[noisy_feature] = 'f'
			elif instance[noisy_feature] == 'f':
				continue
				instance[noisy_feature] = 'm'
			if instance[noisy_feature] == 'in':
				instance[noisy_feature] = 'out'
				continue
			elif instance[noisy_feature] == 'out':
				instance[noisy_feature] = 'in'
				continue
			elif noisy_feature == 'age':
				instance[noisy_feature] = 100 - instance[noisy_feature]
				continue
			elif noisy_feature == 'distance':
				instance[noisy_feature] = 100 - instance[noisy_feature]
				continue
			elif noisy_feature == 'paid_contributions':
				instance[noisy_feature] = 5 - instance[noisy_feature] + 1
				continue
			elif noisy_feature == 'capital':
				instance[noisy_feature] = 10000 - instance[noisy_feature]
				continue
			else:
				instance[noisy_feature] = 1 - float(instance[noisy_feature])
				continue

	conv_db = prep.convertDB(db)
	return db

#Finds how many instances have an incorrect output label after adding noise
def findMistakes(db, discrete):
	output_name = 'satisfied'
	cor_counter = 0
	incor_counter = 0
	for instance in db:
		if (isSatisfied(instance, discrete) and instance[output_name] == 1) or  (not isSatisfied(instance, discrete) and instance[output_name] == 0):
			cor_counter += 1
		else:
			incor_counter += 1
	print('out of ' + repr(len(db)) + ' instance, ' + repr(cor_counter) + ' instances are correct and '+ repr(incor_counter) + ' instances are incorrect, which is ' + repr(100*incor_counter/len(db)) + ' percent incorrect')


#Half of the cases are supposedly incorrect
def isSatisfied(case, discrete):
	if discrete: 
		if case['age'] < 11 and case['gender'] == 0:
			return False
		if case['age'] < 12 and case['gender'] == 1:
			return False
		if case['paid_contributions'] < 4:
			return False
		if case['spouse'] < 1:
			return False
		if case['residence'] < 1:
			return False
		if case['capital'] >= 6:
			return False
		if case['distance'] >= 10 and case['patient_type'] == 0:
			return False
		if case['distance'] < 10 and case['patient_type'] == 1:
			return False
		return True
	else:
		if case['age'] < 0.60 and case['gender'] == 0:
			return False
		if case['age'] < 0.65 and case['gender'] == 1:
			return False
		if case['paid_contributions_1'] == 1:
			return False
		if case['paid_contributions_2'] == 1:
			return False
		if case['paid_contributions_3'] == 1:
			return False
		if case['spouse'] < 1:
			return False
		if case['residence'] < 1:
			return False
		if case['capital'] > 0.29999999999999999:
			return False
		if case['distance'] >= 0.50 and case['patient_type'] == 0:
			return False
		if case['distance'] < 0.50 and case['patient_type'] == 1:
			return False
		return True



############################

if __name__ == "__main__":

	#the level of noise
	noise_level = 0.0

	#The number of cases to generate
	num_cases = 2400

	#The proportion of satisfying cases
	proportion = 0.5

	#The number of noise attributes
	num_noise = 52

	#Generate and export the train database
	db = generateDatabase(num_cases, proportion, num_noise)
	noise_db = addNoiseRegular(db, noise_level=noise_level)
	print(len(db[0]))

	exportCSV(db, 'welfare.csv')
	exportCSV(prep.convertDB(db), 'prepared_welfare.csv')
	exportCSV(prep.discretize(db), 'discrete_welfare.csv')

	exportCSV(noise_db, 'welfare_noise.csv')
	exportCSV(prep.convertDB(noise_db), 'prepared_welfare_noise.csv')
	exportCSV(prep.discretize(noise_db), 'discrete_welfare_noise.csv')

	if False:
		#A mini version of the real one, not used in any experiment
		db = generateDatabase(100, proportion, num_noise)
		print(len(db[0]))
		exportCSV(db, 'welfareMini.csv')
		exportCSV(prep.convertDB(db), 'prepared_welfareMini.csv')
		exportCSV(prep.discretize(db), 'discrete_welfareMini.csv')

		#A larger version of the regular database
		db = generateDatabase(10000, proportion, num_noise)
		exportCSV(db, 'welfareBig.csv')
		print(len(db[0]))
		exportCSV(prep.convertDB(db), 'prepared_welfareBig.csv')
		exportCSV(prep.discretize(db), 'discrete_welfareBig.csv')

	#Generate the test database
	db = generateDatabase(2000, proportion, num_noise)
	noise_db = addNoiseRegular(db, noise_level=noise_level)
	print(len(db[0]))
	
	exportCSV(db, 'welfareTest.csv')
	exportCSV(prep.convertDB(db), 'prepared_welfareTest.csv')
	exportCSV(prep.discretize(db), 'discrete_welfareTest.csv')
	exportCSV(noise_db, 'welfareTest_noise.csv')
	exportCSV(prep.convertDB(noise_db), 'prepared_welfareTest_noise.csv')
	exportCSV(prep.discretize(noise_db), 'discrete_welfareTest_noise.csv')


	#########Age experiment#############

	#The number of cases to generate
	num_cases = 10000

	#The number of noise attributes
	num_noise = 52

	#The steps between the ages of each person
	age_step = 5

	#Generate and export the database
	db = generateAgeDatabase(num_cases, num_noise, age_step, prepared=True)
	noise_db = addNoise(db, noise_level=noise_level)
	print('here')
	print(len(db[0]))
	print(len(noise_db[0]))
	exportCSV(db, 'prepared_ageExperiment.csv')
	exportCSV(noise_db, 'prepared_ageExperiment_noise.csv')
	db = generateAgeDatabase(num_cases, num_noise, age_step, prepared=False)
	noise_db = addNoiseRegular(db, noise_level=noise_level)
	print(len(db[0]))
	exportCSV(db, 'ageExperiment.csv')
	exportCSV(prep.discretize(db), 'discrete_ageExperiment.csv')
	exportCSV(noise_db, 'ageExperiment_noise.csv')
	exportCSV(prep.discretize(noise_db), 'discrete_ageExperiment_noise.csv')


	########Distance experiment ##########

	#The number of cases to generate
	num_cases = 10000

	#The number of noise attributes
	num_noise = 52

	#The steps between the distances
	dist_step = 5

	#Generate and export the database
	db = generateDistDatabase(num_cases, num_noise, dist_step, prepared=True)
	noise_db = addNoise(db, noise_level=noise_level)
	#print('here')
	#print(len(db[0]))
	#print(len(noise_db[0]))
	exportCSV(db, 'prepared_distExperiment.csv')
	exportCSV(noise_db, 'prepared_distExperiment_noise.csv')
	db = generateDistDatabase(num_cases, num_noise, dist_step, prepared=False)
	noise_db = addNoiseRegular(db, noise_level=noise_level)
	print(len(db[0]))
	exportCSV(db, 'distExperiment.csv')
	exportCSV(prep.discretize(db), 'discrete_distExperiment.csv')
	exportCSV(noise_db, 'distExperiment_noise.csv')
	exportCSV(prep.discretize(noise_db), 'discrete_distExperiment_noise.csv')


	######## One fail feature ##########

	#The number of cases to generate
	num_cases = 2400

	#The proportion of satisfying cases
	proportion = 0.5

	#The number of noise attributes
	num_noise = 52

	#Generate and export the database
	db = generateOneFailDatabase(num_cases, proportion, num_noise)
	noise_db = addNoiseRegular(db, noise_level=noise_level)
	print(len(db[0]))
	exportCSV(db, 'welfare_onefail.csv')
	exportCSV(prep.convertDB(db), 'prepared_welfare_onefail.csv')
	exportCSV(prep.discretize(db), 'discrete_welfare_onefail.csv')
	exportCSV(noise_db, 'welfare_onefail_noise.csv')
	exportCSV(prep.convertDB(noise_db), 'prepared_welfare_onefail_noise.csv')
	exportCSV(prep.discretize(noise_db), 'discrete_welfare_onefail_noise.csv')

	#the one fail test set
	db = generateOneFailDatabase(2000, proportion, num_noise)
	noise_db = addNoiseRegular(db, noise_level=noise_level)
	print(len(db[0]))
	exportCSV(db, 'welfareTest_onefail.csv')
	exportCSV(prep.convertDB(db), 'prepared_welfareTest_onefail.csv')
	exportCSV(prep.discretize(db), 'discrete_welfareTest_onefail.csv')
	exportCSV(noise_db, 'welfareTest_onefail_noise.csv')
	exportCSV(prep.convertDB(noise_db), 'prepared_welfareTest_onefail_noise.csv')
	exportCSV(prep.discretize(noise_db), 'discrete_welfareTest_onefail_noise.csv')


