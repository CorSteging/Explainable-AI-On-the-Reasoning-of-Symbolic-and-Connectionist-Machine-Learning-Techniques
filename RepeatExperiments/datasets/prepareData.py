#Converts a data set into a data set that works in a NN.
#Author: Cor Steging

import sys
import csv
import pandas

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

#Checks if a variable is a number
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


#Converts a database into a NN-ready database
def convertDB(oldDB):

	#If a feature has less than this amount of possible values, it is a categorical feature
	categorical_threshold = 10

	df = pandas.DataFrame(oldDB).apply(pandas.to_numeric, errors="ignore")
	newDB = df

	for column in df:

		#Should the column be deleted
		delete = False

		#print(column, df[column].dtype)
		newColumn = []

		#If the data is numeric:
		if(df[column].dtype == 'int64' or df[column].dtype == 'float64'):
			values = df[column].unique()
			num_values = len(values)
			min_value = df[column].min()
			max_value = df[column].max()

			#If the feature has the right values, continue
			if(min_value == 0 and max_value == 1):
				newColumn = df[column]

			#If the number of values is 2, but they are not 0 and 1, set them to 0 and 1
			elif(num_values == 2):

				for instance in df[column]:
				#Change male to 1, female to 0, out to 1, in to 0
					if instance == values[0]: newColumn.append(0)
					if instance == values[1]: newColumn.append(1)
					if instance == 'm': newColumn.append(1)
					if instance == 'f': newColumn.append(0)
					if instance == 'out': newColumn.append(1)
					if instance == 'in': newColumn.append(0)

			#If it represents a categorical feature
			elif(num_values < categorical_threshold):
				#Delete the original column
				delete = True

				#Create new columns based on the number of values
				for x in range(0, num_values):
					temp_column = []
					for instance in df[column]:
						if instance == values[x]:
							temp_column.append(1)
						else:
							temp_column.append(0)
					newDB[column+'_'+str(values[x])] = temp_column

			#Normalize the values between 0 and 1
			else:
				for instance in df[column]:
					newColumn.append(round((instance - min_value)/(max_value - min_value), 3))
		
		#If the data is non-numeric
		if(df[column].dtype == 'object'):
			values = df[column].unique()
			num_values = len(values)

			#If there are only two possible values, appoint 0 and 1 to them
			if(num_values == 2):
				for instance in df[column]:
					if instance == values[0]: newColumn.append(0)
					if instance == values[1]: newColumn.append(1)
			else:
				#Delete the original column
				delete = True

				#Create new columns based on the number of values
				for x in range(0, num_values):
					temp_column = []
					for instance in df[column]:
						if instance == values[x]:
							temp_column.append(1)
						else:
							temp_column.append(0)
					newDB[column+'_'+values[x]] = temp_column
	
		#Add the new column to the new database or delete it
		if(delete):
			del newDB[column]
		else:
			newDB[column] = newColumn

	return newDB.to_dict('records')


#Converts a database into a db with only discrete values (1 or 0) for association rules
def discretize(oldDB, num_bins=20):

	df = pandas.DataFrame(oldDB).apply(pandas.to_numeric, errors="ignore")
	newDB = df


	for column in df:

		#Should the column be deleted
		delete = False

		#print(column, df[column].dtype)
		newColumn = []

		#If the data is numeric:
		if(df[column].dtype == 'int64' or df[column].dtype == 'float64'):
			values = df[column].unique()
			num_values = len(values)
			min_value = df[column].min()
			max_value = df[column].max()

			#If the feature has the right values, continue
			if(min_value == 0 and max_value == 1):
				newColumn = df[column]

			#If the number of values is 2, but they are not 0 and 1, set them to 0 and 1
			elif(num_values == 2):
				for instance in df[column]:
					if instance == values[0]: newColumn.append(0)
					if instance == values[1]: newColumn.append(1)
					if instance == 'm': newColumn.append(1)
					if instance == 'f': newColumn.append(0)
					if instance == 'out': newColumn.append(1)
					if instance == 'in': newColumn.append(0)

			#If it represents a categorical feature
			elif(num_values < num_bins):

				#Check if the values are numerical
				numerical = True
				for instance in df[column]:
					if not isNumber(instance):
						numerical = False
						break

				#If all values are numerical, continue
				if numerical == True:
					newColumn = df[column]
				else:
					for instance in df[column]:
						newColumn.append(list(values).index(instance))

			#Else, data is numerical. So discretize them
			else:
				bin_size = round((max_value-min_value)/num_bins)

				for instance in df[column]:
					remainder = instance%bin_size
					newColumn.append((instance-remainder)/bin_size)


		
		#If the data is non-numeric
		if(df[column].dtype == 'object'):
			values = df[column].unique()
			num_values = len(values)

			#If there are only two possible values, appoint 0 and 1 to them
			if(num_values == 2):
				for instance in df[column]:
					if instance == values[0]: newColumn.append(0)
					if instance == values[1]: newColumn.append(1)
			else:
				for instance in df[column]:
					newColumn.append(list(values).index(instance))
	
		#Add the new column to the new database or delete it
		if(delete):
			del newDB[column]
		else:
			newDB[column] = newColumn

	return newDB.to_dict('records')


if __name__ == "__main__":
	#Read the original database
	oldDB = readCSV(sys.argv[1])

	#Convert the old database
	newDB = convertDB(oldDB)

	#Export the new database
	exportCSV(newDB, 'prepared_'+sys.argv[1])	

	#discretisze the old database
	newDB = discretize(oldDB)

	#Export the new database
	exportCSV(newDB, 'discrete_'+sys.argv[1])