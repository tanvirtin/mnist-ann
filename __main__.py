import math
from mnist import MNIST
from NeuralNetwork import NeuralNetwork
from tqdm import tqdm
import scipy.special

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

def main():
	
	mndata = MNIST("./data")

	# data to train
	images_train, labels_train = mndata.load_training()	

	# data to test
	images_test, labels_test = mndata.load_testing()

	# one hot encoding the labels
	labels_train = processLabels(list(labels_train))
	labels_test = processLabels(list(labels_test))

	nn = NeuralNetwork([784, 200, 10], 0.01, lambda x: scipy.special.expit(x), lambda x: x * (1 - x))

	epochs = 5

	for i in range(epochs):
		print("Epoch number", i + 1)
		for j in tqdm(range(len(images_train))):
			nn.train(images_train[j], labels_train[j])

	for i in range(10):
		print(nn.query(images_test[i]))
		print(labels_test[i])




if __name__ == "__main__":
	main()