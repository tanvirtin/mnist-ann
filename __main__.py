import math
from mnist import MNIST
from NeuralNetwork import NeuralNetwork
from tqdm import tqdm
import scipy.special
import random

def oneHotEncoding(num):
	encoded = []

	for i in range(10):
		encoded.append(0)

	encoded[num] = 1;

	return encoded

def processLabels(labels):
	for i in range(len(labels)):
		labels[i] = oneHotEncoding(labels[i])

	return labels

def kFoldsPreperation(images_train, labels_train):
	dataPerFold = int(len(images_train) / 10)

	imageFolds = []

	labelFolds = []

	iFold = []
	lFold = []
	for i in range(len(images_train)):
		iFold.append(images_train[i])
		lFold.append(labels_train[i])
		# if i is a multiple of the datafold then we add the fold to the folds layer
		# because only a number divisible by 60000 means it is a multiple of 60000
		# like 60000 * 2, 60000 * 3, etc
		# 0 mod anything is 0 so I check for that
		if i != 0 and i % dataPerFold == 0:
			# we append the fold to the folds arrays
			imageFolds.append(iFold)
			labelFolds.append(lFold)
			# and we empty out the iFold and lFold array for new dataset to be filled
			iFold = []
			lFold = []

	return (imageFolds, labelFolds)

def train(nn, imageFolds, labelFolds):
	# pick a random number, the index that gets tested that round won't be tested again ever
	# and gets placed in an array of indexes that has already been tested
	indexesTested = [] # contains the indexes from the folds array for which data has been tested

	# will contain values 0 to imageFolds
	pool = []
	for i in range(len(imageFolds)):
		pool.append(i)

	# holds the array of all the percentages that were obtained while training
	percentageList = []

	# we loop till the indexesTested is not equal to 10
	while len(indexesTested) != len(imageFolds):
		# selects a random number from a pool of 0 to 9 values
		poolIndex = random.randint(0, len(pool) - 1)
		randomIndex = pool[poolIndex]
		# after that specific value is picked from 0 to 9 we delete the value from pool so that it cannot be
		# selected again
		del pool[poolIndex]

		print("Training on folds....")
		for i in range(len(imageFolds)):
			# if i equals the randomIndex we skip the loop
			if i == randomIndex:
				continue
			for j in tqdm(range(len(imageFolds[i]))):
				nn.train(imageFolds[i][j], labelFolds[i][j])


		# now we test the randomIndex selected from the k folds
		print("Testing on a random fold at index {}....".format(randomIndex))
		accuracy = 0
		for i in tqdm(range(len(imageFolds[randomIndex]))):
			result = nn.query(imageFolds[randomIndex][i])
			result = result.tolist()

			if (result.index(max(result)) == labelFolds[randomIndex][i].index(max(labelFolds[randomIndex][i]))):
				accuracy += 1

		# at the end of the for loop we find the accuracy in percentage

		percentageAccuracy = (accuracy / len(imageFolds[randomIndex])) * 100

		print("Accuracy of Neural Network at the moment... {}%".format(percentageAccuracy))

		indexesTested.append(randomIndex)

		percentageList.append(percentageAccuracy)

	summation = 0
	for i in range(len(percentageList)):
		summation += percentageList[i]

	mean = summation / len(percentageList)

	print("Overall accuracy of the Neural Network is... {}%".format(mean))


def main():
	mndata = MNIST("./data")
	# data to train
	images_train, labels_train = mndata.load_training()	
	# data to test
	images_test, labels_test = mndata.load_testing()

	# one hot encoding the labels
	labels_train = processLabels(list(labels_train))
	labels_test = processLabels(list(labels_test))

	imageFolds, labelFolds = kFoldsPreperation(images_train, labels_train)


	print("Training ANN with 1 hidden layer with 200 HU...")
	nn = NeuralNetwork([784, 200, 10], 0.01, 0.001, lambda x: scipy.special.expit(x), lambda x: x * (1 - x))

	train(nn, imageFolds, labelFolds)

	print("Training ANN with 1 hidden layer with 500 HU...")
	nn = NeuralNetwork([784, 500, 10], 0.01, 0.001, lambda x: scipy.special.expit(x), lambda x: x * (1 - x))

	train(nn, imageFolds, labelFolds)

	print("Training ANN with 3 hidden layers with 1300 HU...")
	nn = NeuralNetwork([784, 300, 300, 10], 0.01, 0.001, lambda x: scipy.special.expit(x), lambda x: x * (1 - x))

	train(nn, imageFolds, labelFolds)





if __name__ == "__main__":
	main()