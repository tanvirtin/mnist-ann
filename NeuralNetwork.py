import numpy as np
import math

class NeuralNetwork(object):
	# The neural network will take an array as its constructor
	# this array will determine how many layers the neural network
	# have and how many neurons will be in each layer. The length
	# of the dimensions will be the number of layers the Neural
	# Network will have.

	def __init__(self, dimensions, activationFunction, activationPrimeFunction):
		self.learningRate = 0.01
		self.weights = []
		self.outputs = []
		# key thing is that number of weights attached to the current layer's each neuron
		# is equal to the number of neurons in the previous layer
		for i in range(1, len(dimensions)):
			layer = []
			for j in range(dimensions[i]):
				layer.append(np.random.rand(dimensions[i - 1]))
			layer = np.array(layer)
			self.weights.append(layer)

		# transform the weights multidimensional array into a multidimensional numpy array
		self.weights = np.array(self.weights)

		# activation function
		self.f = activationFunction
		# differentiation of the activation function
		self.fPrime = activationPrimeFunction 


	# Feed forward needs an array of inputs which will get multiplied
	# and passed down from layer to layer representing different outputs
	def feedForward(self, inputArray):
		outputs = []
		inputArray = np.array(inputArray)
		self.inputArray = inputArray

		# each iteration in this for loop will represent a layer in a neural network
		# we start from the second layer in the neural network because the second layer weights
		# connect to the first layer thats why
		for i in range(len(self.weights)):
			output = []

			# each iteration in this for loop will represent a neuron in this particular layer
			# as the weight multi dimensional arrays each index contains layers and inside that
			# each index will represent each neuron in a network
			for j in range(len(self.weights[i])):
				if (i == 0):
					output.append(self.f(np.dot(self.weights[i][j], inputArray)))
				else:
					output.append(self.f(np.dot(self.weights[i][j], outputs[i - 1])))

			output = np.array(output)
			outputs.append(output)

		outputs = np.array(outputs)
		self.outputs = outputs


	def backPropagation(self, target):
		target = np.array(target)
		weightChanges = []
		# holds the delta values that are needed for other layers
		# will have the same number elements as outputs
		deltas = [None] * len(self.outputs)
		# for loop will break when the i will get to -1
		# and the for loop will start from len(self.outputs) - 1
		# its exactly like a reverse for loop in other language where
		# i starts from len(self.output) - 1 and goes on till when i > 0
		for i in reversed(range(len(self.outputs))):
			weightChange = [] # weight change in each layer
			# when i is equal to the length - 1 of the array we are at the very
			# last layer
			if i == len(self.outputs) - 1:
				# self.outputs[i] because we are dealing with the output of the last layer
				# which is essentially the output of the neurons in the output layer
				e = self.eOutput(target, self.outputs[i])
				delta = self.delta(e, self.fPrime(self.outputs[i]))
				# push the delta array to the array of delta arrays in each layer
				deltas[i] = delta

				# after we get the delta the number of deltas we will get in each
				# layer is equal to the number of neurons in that layer

				# loops over the delta array which is the number of neurons in that layer
				for j in range(len(delta)):
					# we declare a weight array for each neuron in this layer
					# as number of deltas equals number of neurons in that layer
					weightChangePerNeuron = []
					# length of the outputs array at the specific index will give you the 
					# number of neurons in that specific layer
					# this loop will loop over the neurons in the previous layer
					for k in range(len(self.outputs[i - 1])):
						weightChangePerNeuron.append(delta[j] * self.outputs[i - 1][k])
					weightChangePerNeuron = np.array(weightChangePerNeuron)
					weightChange.append(weightChangePerNeuron)
			else:
				# backpropagation formula works differently when you are in the hidden layers
				# e should be an array containing the errors
				e = self.eJ(deltas, self.weights, i)
				delta = self.delta(e, self.fPrime(self.outputs[i]))
				# assign the delta array theo the array of delta arrays in each layer
				deltas[i] = (delta)

				# i - 1 will result in an error as it will give you the element in the last layer
				# if i is - 1 we are dealing with the input layer

				for j in range(len(delta)):
					weightChangePerNeuron = []
					# if i - 1 == -1 it means that we are in the second layer of the neural network
					# which means that the output in the next layer should be our input layer
					if i - 1 == -1:
						for k in range(len(self.inputArray)):
							# number of neurons in the input layer is equal to i
							weightChangePerNeuron.append(delta[j] * self.inputArray[i])
						weightChangePerNeuron = np.array(weightChangePerNeuron)
					else :
						for k in range(len(self.outputs[i - 1])):
							weightChangePerNeuron.append(delta[j] * self.outputs[i - 1][j])
						weightChangePerNeuron = np.array(weightChangePerNeuron)

					weightChange.append(np.array(weightChangePerNeuron))

			weightChange = np.array(weightChange)

			weightChanges.append(weightChange)

		weightChanges = np.array(weightChanges)

		# reversing the weightChanges array
		weightChanges = weightChanges[::-1]

		# updatating the weights now
		self.weights = self.weights + (self.learningRate * weightChanges)


	# finds the e in the output of the neural 
	def eOutput(self, target, output):
		return np.subtract(target, output)

	# to find ej we need the delta's of the layer right next to it and the weights
	# from the next layer that connect to the current neuron for which the e we find
	def eJ(self, deltas, weights, layer):
		e = []
		# this for loop loops over the weights array in a specific layer specified by the layer variable
		# the elements in the weights array are arrays containing one or many weights, and each element can 
		# represent the number of neurons in the array
		for i in range(len(weights[layer])):
			# variable to store eJ
			eJ = 0
			# this loop iterates over the next layer's neurons or the layer right next to it
			# we use deltas because deltas elements represent layers in the neural network and inside
			# that array we have the number of neurons in a specific layer
			for j in range(len(deltas[layer + 1])):
				# Each neuron in this layer has weights coming into it as weights come from right to left in a network
				# what happens here is that the weight contributing or attached to the neuron in this layer is equal to the
				# number of neurons in the layer right next to it. So what happens is that we multiply the weights which are attached
				# to this neuron by the delta associated with that weight in the layer right next to it
				# This is the trickiest concept in backpropagation
				eJ += deltas[layer + 1][j] * weights[layer + 1][j][i] # its weightsK[layers + 1][j][i] because index of the weight array depends on the neuron we are in the previous layer

			e.append(eJ)

		return e
		
	def delta(self, eJ, fPrimeJ):
		delta = eJ * fPrimeJ
		return delta

	def displayLayers(self):
		print(self.weights)

	def displayOutputs(self):
		print(self.outputs)

def main():
	nn = NeuralNetwork([1, 3, 2], lambda x: (1 / (1 + math.exp(-x))), lambda x: (x * (1 - x)))
	
	print("")

	nn.feedForward([1])

	nn.backPropagation([2, 2]);


if __name__ == "__main__":
	main()