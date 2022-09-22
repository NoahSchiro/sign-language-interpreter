import csv
# We don't care all that much about a test train 
# split, so we are going to just merge our data 
# sets together to get maximum amounts of training data

trainData = "resources/train/train.csv"
testData  = "resources/test/test.csv"

# We will append test to train

# Open this in append mode
with open(trainData, 'a') as train:
	
	# Open this in read
	with open(testData, 'r') as test:

		testReader = csv.reader(test)
		trainWriter = csv.writer(train)

		# Read in each row
		for row in testReader:
			
			# Write to train
			trainWriter.writerow(row)


	# Close files
	test.close()	
train.close()
