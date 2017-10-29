from tqdm import tqdm
import numpy
import math
import scipy.special

# neural network class definition
class NeuralNetwork:

	# initialise the neural network
	def __init__(self, dimensions, learningRate, weightDecay, activationFunction, activationFunctionPrime):
		# # this array will contain all the layers of the neural network
		self.layers = []

		# we construct the hidden layers
		for i in range(1, len(dimensions)):
			hiddenLayer = []
			# number of weight arrays will be determined by the number of neurons
			# in the current layer 
			for j in range(dimensions[i]):
				# number of weights per neuron is equal to the number of nodes
				# in the previous layer
				# very important for the weights to be assigned with random values from -1 to +1
				# as our algorithm adds up the change in weight
				hiddenLayer.append(numpy.random.uniform(-1, 1, dimensions[i - 1]))

			# make the hiddenLayer into a numpy array
			hiddenLayer = numpy.array(hiddenLayer)

			self.layers.append(hiddenLayer)

		# learning rate
		self.learningRate = learningRate

		# weight decay rate
		self.weightDecay = weightDecay
		
		# activation function
		self.f = activationFunction
		# differentiated activation function
		self.fPrime = activationFunctionPrime

	def backPropagation(self, inputs, targets):
		# conversion of inputs and target arrays to transposed numpy matrixes
		inputs = numpy.transpose(numpy.array([numpy.array(inputs)]))
		targets = numpy.transpose(numpy.array([numpy.array(targets)]))

		# numpy array of outputs in each layer retrieved
		outputs = self.feedForward(inputs)

		# the error of each neuron in the output layer
		outputsError = targets - outputs[len(outputs) - 1]

		# the array of  errors of each neuron in each layer
		hiddenErrors = [None] * (len(outputs) - 1)

		# we loop backwards excluding the last layer as the hiddenlayer has a size of number of weight layers - 1
		for i in reversed(range(len(outputs) - 1)):
			hiddenError = 0
			if i == len(outputs) - 2:
				hiddenError = numpy.dot(numpy.transpose(self.layers[i + 1]), outputsError)
			else:
				hiddenError = numpy.dot(numpy.transpose(self.layers[i + 1]), hiddenErrors[i + 1])

			hiddenErrors[i] = hiddenError

		# update the weights using weight decay
		for i in reversed(range(len(self.layers))):
			if i == len(self.layers) - 1:
				self.layers[i] += (self.learningRate * numpy.dot((outputsError * self.fPrime(outputs[i])), numpy.transpose(outputs[i - 1]))) + (self.learningRate * self.weightDecay * self.layers[i])
			elif i == 0:
				self.layers[i] += (self.learningRate * numpy.dot((hiddenErrors[i] * self.fPrime(outputs[i])), numpy.transpose(inputs))) + (self.learningRate * self.weightDecay * self.layers[i])
			else:
				self.layers[i] += (self.learningRate * numpy.dot((hiddenErrors[i] * self.fPrime(outputs[i])), numpy.transpose(outputs[i - 1]))) + (self.learningRate * self.weightDecay * self.layers[i])

	def train(self, inputs, targets):
		self.backPropagation(inputs, targets)

	def feedForward(self, inputs):
		# will contain arrays of all the outputs in each layer of the neural network
		outputs = []

		# this will loop over the layers of weights
		for i in range(len(self.layers)):
			# if we are in the first layer we are dealing with the inputs provided
			# to the neural network
			if i == 0:
				output = self.f(numpy.dot(self.layers[i], inputs))
			# else we are dealing with the output of the hidden layers
			else:
				output = self.f(numpy.dot(self.layers[i], outputs[i - 1]))

			output = numpy.array(output)
			# we finally append the hiddenOutput to the layers of hiddenOutputs
			outputs.append(output)

		return outputs


	# query the neural network
	def query(self, inputs):
		# convert inputs list to 2d array
		inputs = numpy.transpose(numpy.array([numpy.array(inputs)]))
		return self.feedForward(inputs)[-1]


def main():
	nn = NeuralNetwork([2, 2, 1], 0.01, 0.001, lambda x: scipy.special.expit(x), lambda x: x * (1 - x))

	for i in tqdm(range(1000000)):
		nn.train([0, 0], [0])
		nn.train([0, 1], [0])
		nn.train([1, 1], [1])

	print(nn.query([0, 1]))
	print(nn.query([1, 1]))
	print(nn.query([1, 0]))
	print(nn.query([0, 0]))


	

if __name__ == "__main__":
	main()
