import numpy as np
import math

class NeuralNetwork(object):
	# The neural network will take an array as its constructor
	# this array will determine how many layers the neural network
	# have and how many neurons will be in each layer. The length
	# of the dimensions will be the number of layers the Neural
	# Network will have.

	def __init__(self, dimensions, activationFunction, activationPrimeFunction):
		self.weights = []
		self.outputs = []
		# key thing is that number of weights attached to the current layer's each neuron
		# is equal to the number of neurons in the previous layer
		for i in range(1, len(dimensions)):
			layer = []
			for j in range(dimensions[i]):
				layer.append(np.random.rand(dimensions[i - 1]))
			self.weights.append(layer)

		# activation function
		self.f = activationFunction
		# differentiation of the activation function
		self.fPrime = activationPrimeFunction 

	# Feed forward needs an array of inputs which will get multiplied
	# and passed down from layer to layer representing different outputs
	def feedForward(self, inputArray):
		outputs = []
		inputArray = np.array(inputArray)

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
		print(deltas)
		# for loop will break when the i will get to -1
		# and the for loop will start from len(self.outputs) - 1
		# its exactly like a reverse for loop in other language where
		# i starts from len(self.output) - 1 and goes on till when i > 0
		for i in reversed(range(len(self.outputs))):
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
					weightChange = []
					# length of the outputs array at the specific index will give you the 
					# number of neurons in that specific layer
					# this loop will loop over the neurons in the previous layer
					for k in range(len(self.outputs[i - 1])):
						weightChange.append(delta[j] * self.outputs[i - 1][k])

					weightChanges.append(np.array(weightChange))

			else:
				# e should be an array containing the errors
				e = self.eJ(deltas, self.weights, i)
				
				delta = self.delta(e, self.fPrime(self.outputs[i]))
				# push the delta array to the array of delta arrays in each layer
				deltas[i] = (delta)

				for j in range(len(delta)):
					weightChange = []

					for k in range(len(self.outputs[i - 1])):
						weightChange.append(delta[j] * self.outputs[i - 1][k])



		print(weightChanges)




	# finds the e in the output of the neural 
	def eOutput(self, target, output):
		return np.subtract(target, output)

	# to find ej we need the delta's of the layer right next to it and the weights
	# from the next layer that connect to the current neuron for which the e we find
	def eJ(self, deltas, weights, layer):
		e = []
		# this array loops through the current layer neuron times
		for i in range(len(weights[layer])):
			eJ = 0
			# this loop iterates over the next layer's neurons
			for j in range(len(deltas[layer + 1])):
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
	nn = NeuralNetwork([1, 2, 2], lambda x: (1 / (1 + math.exp(-x))), lambda x: (x * (1 - x)))
	
	print("")

	nn.feedForward([1])

	nn.displayOutputs()

	nn.backPropagation([2, 2]);


if __name__ == "__main__":
	main()