import numpy as np

class NeuralNetwork(object):
	# The neural network will take an array as its constructor
	# this array will determine how many layers the neural network
	# have and how many neurons will be in each layer. The length
	# of the dimensions will be the number of layers the Neural
	# Network will have.
	def __init__(self, dimensions):
		self.weights = []
		for i in range(1, len(dimensions)):
			layer = []
			for j in range(dimensions[i]):
				layer.append(np.random.rand(dimensions[i - 1]))
			self.weights.append(layer)

	# Feed forward needs an array of inputs which will get multiplied
	# and passed down from layer to layer representing different outputs
	def feedForward(self, inputArray):
		outputs = []
		inputArray = np.asarray(inputArray)

		# each indexes in weights array represents layers in a neural network
		for i in range(len(self.weights)):
			output = []
			# indexes of each layer array inside the weights array represents
			# the weight array for that neuron, where each index is an array of weights
			# for each neuron
			for j in range(len(self.weights[i])):
				if (i == 0):
					output.append(np.dot(self.weights[i][j], inputArray))
				else:
					output.append(np.dot(self.weights[i][j], outputs[i - 1]))

			output = np.asarray(output)

			outputs.append(output)

		return outputs


	def backPropagation(self):
		pass

	def displayLayers(self):
		print(self.weights)



def main():
	nn = NeuralNetwork([1, 2, 1])
	nn.displayLayers()

	print("")

	print(nn.feedForward([3]))


if __name__ == "__main__":
	main()