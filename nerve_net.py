from random import random
from math import exp
from random import seed

##########################
# Network Initialization #
##########################

def _initialize_layer(network: dict, layer, layer_neurons: int) -> None:
	'''
	Initializes a layer of the network with the correct number of weights,
	biases, outputs, and errors based on its number of neurons.
	'''
	#Neurons in layer.
	network["neuron_counts"][layer] = layer_neurons
	#Each element is a given neurons output.
	network["outputs"][layer] = [0.0] * layer_neurons
	#Input layer neurons only need a outputs.
	if layer == 0:
		return
	#Each element is a given neurons bias.
	network["biases"][layer] = [0.0] * layer_neurons
	#Each element is a given neurons error.
	network["errors"][layer] = [0.0] * layer_neurons
	#Each element is a given neurons accumulated error.
	network["accumulated_errors"][layer] = [0.0] * layer_neurons
	#Each row is a given neurons weights.
	network["weights"][layer] = [[]] * layer_neurons
	for neuron in range(layer_neurons):
		#Weights per neuron is equal to prior layers number of neurons.
		prior_layer_neurons = network["neuron_counts"][layer - 1]
		network["weights"][layer][neuron] = [0.0] * prior_layer_neurons
	#Initialize layers wieghts and biases to small random values.
	for neuron in range(layer_neurons):
		network["biases"][layer][neuron] = random()
		for weight in range(network["neuron_counts"][layer - 1]):
			network["weights"][layer][neuron][weight] = random()

def create_network(input_neurons: int, hidden_neurons: int, hidden_layers: int, output_neurons: int) -> dict:
	'''
	Returns an untrained network with the given layer structure. Also sets
	default learning parameters.
	'''
	#Create network dictionary.
	network = {"layer_count": hidden_layers + 2}
	#Vectors where each element is data for a given layer.
	network["neuron_counts"] = [0.0] * network["layer_count"]
	network["outputs"] = [[]] * network["layer_count"]
	network["biases"] = [[]] * network["layer_count"]
	network["errors"] = [[]] * network["layer_count"]
	network["accumulated_errors"] = [[]] * network["layer_count"]
	network["weights"] = [[]] * network["layer_count"]
	#Initialize network layers.
	_initialize_layer(network, 0, input_neurons)
	for layer in range(1, hidden_layers + 1):
		_initialize_layer(network, layer, hidden_neurons)
	_initialize_layer(network, -1, output_neurons)
	#Set defaults for training.
	network["learning_rate"] = 0.20
	network["epochs"] = 512
	network["activation_function"] = _sigmoid_activation
	network["activation_derivative"] = _sigmoid_derivative
	network["learning_type"] = _learn_online
	return network

def set_learning_rate(network: dict, learning_rate: float) -> None:
	'''
	Sets the learning rate that will be used during train_network().
	The learning rate controls how much the weights and bias will be
	chnaged to correct for neuron errors. Number should be less than one.
	'''
	network["learning_rate"] = learning_rate

def set_epochs(network: dict, epochs: int) -> None:
	'''
	Sets the number of epochs that will be ran during train_network().
	An epoch is training the network on the given data set once. 42 epochs
	would mean training the network on the given data set 42 times.
	'''
	network["epochs"] = epochs

def set_activation_function(network: dict, function_name: str) -> None:
	'''
	Sets the activation function to be used by the networks neurons during
	train_network().
	The activation function is used during forward propagation when
	calculating a given neurons output (activation). Activation functions
	supported:
		"sigmoid" - Classic activation functions.
		"tanh" - Most the time better then sigmoid.
	'''
	#Use function for forward propagation and its derivative for backward propagation.
	match function_name:
		case "sigmoid":
			network["activation_function"] = _sigmoid_activation
			network["activation_derivative"] = _sigmoid_derivative
		case "tanh":
			network["activation_function"] = _tanh_activation
			network["activation_derivative"] = _tanh_derivative

def set_learning_type(network: dict, learning_name: str) -> None:
	'''
	Set the learning type that will be used during train_network().
	The learning type dictates when the networks weights and bias are
	updated. Learning types supported:
		"online" - weights and bias updated for every training example.
		"batch" - weights and bias updated for every epoch.
	'''
	match learning_name:
		case "online":
			network["learning_type"] = _learn_online
		case "batch":
			network["learning_type"] = _learn_batch

###############################
# Network Forward Propagation #
###############################

def _sigmoid_activation(summation: float) -> float:
	'''
	Sigmoid function formula:
	f(x) = 1/(1 + exp(-x))
	'''
	output = 1.0 / (1.0 + exp(-summation))
	return output

def _tanh_activation(summation: float) -> float:
	'''
	Hyperbolic tangent function formula:
	f(x) = tanh(x) = 2/(1 + exp(-2x)) - 1
	'''
	output = (2 / (1 + exp(-2 * summation))) - 1
	return output

def _neuron_output(network: dict, layer: int, neuron: int) -> float:
	'''
	Neuron activation formula:
	activation_function(sum(weight_i * input_i) + bias)
	where:
		weight_i is the weight that connects the current neuron to the prior layers i'th nueron.
		input_i is the prior layers i'th nuerons activation.
		bias is the current neurons bias.
		activation_function is sigmoid, relu, etc...
	'''
	summation = network["biases"][layer][neuron]
	for index in range(network["neuron_counts"][layer - 1]):
		input = network["outputs"][layer - 1][index]
		weight = network["weights"][layer][neuron][index]
		summation += input * weight
	output = network["activation_function"](summation)
	return output

def _forward_propagate(network: dict, input_values: list[float]) -> None:
	'''
	Perform forward propagation on the given network.
	This means calculating the outputs (activations) for each of the
	networks neurons. This is done in layer by layer starting at the
	input layer.
	'''
	#Outputs for input layer neurons are the input values themselves.
	for neuron in range(network["neuron_counts"][0]):
		output = input_values[neuron]
		network["outputs"][0][neuron] = output
	#Propagate outputs forward, layer by layer, neuron by neuron.
	for layer in range(1, network["layer_count"]):
		for neuron in range(network["neuron_counts"][layer]):
			output = _neuron_output(network, layer, neuron)
			network["outputs"][layer][neuron] = output

################################
# Network Backward Propagation #
################################

def _sigmoid_derivative(summation: float) -> float:
	'''
	Sigmoid derivative formula:
	f'(x) = x * (1 - x)
	'''
	error = summation * (1.0 - summation)
	return error

def _tanh_derivative(summation: float) -> float:
	'''
	Hyperbolic tangent derivative formula:
	f'(x) = 1 - x^2
	'''
	output = 1 - (summation**2)
	return output

def _output_neuron_error(network: dict, expected_values: list[float], neuron: int) -> float:
	'''
	Output neuron error formula:
	error = (output - expected) * activation_derivative(output)
	where:
		Expected is the expected output (activation) value for the
			neuron.
		output is the current output (activation) value for the neuron.
		activation_derivative is sigmoid, relu, etc...
	'''
	expected = expected_values[neuron]
	output = network["outputs"][-1][neuron]
	error = (output - expected) * network["activation_derivative"](output)
	return error

def _hidden_neuron_error(network: dict, layer: int, neuron: int) -> float:
	'''
	NOTE: think of this formula as neuron activation but going backwards.
		With errors as inputs instead of activations.
		And activation derivatives instead of activation functions.
	Hidden neuron error formula:
	error = (weight_k * error_j) * activation_derivative(output)
	Where:
		weight_k is the weight that connects the current neuron to the
			next layers k'th nueron.
		error_j is the next layers j'th nuerons error (neuron at the
			other end of the weight_k).
		output is the current output (activation) value for the neuron.
		activation_derivative is sigmoid, relu, etc...
	'''
	summation = 0.0
	for index in range(network["neuron_counts"][layer + 1]):
		error = network["errors"][layer + 1][index]
		weight = network["weights"][layer + 1][index][neuron]
		summation += error * weight
	output = network["outputs"][layer][neuron]
	error = summation * network["activation_derivative"](output)
	return error

def _backward_propagate(network: dict, expected_values: list[float]) -> None:
	'''
	Perform backward propagation on the given network.
	This means calculating the errors for each of the networks neurons.
	This is done in layer by layer starting at the output layer.
	'''
	#Errors for output layer neurons are calculated using expected values.
	for neuron in range(network["neuron_counts"][-1]):
		error = _output_neuron_error(network, expected_values, neuron)
		network["errors"][-1][neuron] = error
	#Propagate errors backward, layer by layer, neuron by neuron.
	for layer in range(-2, -(network["layer_count"]), -1):
		for neuron in range(network["neuron_counts"][layer]):
			error = _hidden_neuron_error(network, layer, neuron)
			network["errors"][layer][neuron] = error

####################
# Network Updating #
####################

def _adjusted_bias(network: dict, layer: int, neuron: int) -> float:
	'''
	Adjusted bias formula:
	bias = bias - learning_rate * error
	Where:
		bias is a current bias.
		learning_rate is a user parameter.
		error is the error calculated by the backpropagation procedure
			(neurons error).
	'''
	current_bias = network["biases"][layer][neuron]
	error = network["errors"][layer][neuron]
	new_bias = current_bias - network["learning_rate"] * error
	return new_bias

def _adjusted_weight(network: dict, layer: int, neuron: int, weight: int) -> float:
	'''
	Adjusted weight formula:
	weight = weight - learning_rate * error * input
	Where:
		weight is a current weight.
		learning_rate is a user parameter.
		error is the error calculated by the backpropagation procedure
			(neurons error).
		input is the value that caused the error (prior layers nuerons
			activation, neuron at the other end of the weight).
	'''
	current_weight = network["weights"][layer][neuron][weight]
	error = network["errors"][layer][neuron]
	input = network["outputs"][layer - 1][weight]
	new_weight = current_weight - network["learning_rate"] * error * input
	return new_weight

def _adjust_network(network: dict) -> None:
	'''
	Adjusts each of the networks neurons weights and bias. Adjustments
	based on the given nuerons error (error calculated during backward
	propagation).
	'''
	#Update networks biases and weights, layer by layer, neuron by neuron.
	for layer in range(1, network["layer_count"]):
		for neuron in range(network["neuron_counts"][layer]):
			#Calculate a new bias for the current neuron.
			new_bias = _adjusted_bias(network, layer, neuron)
			network["biases"][layer][neuron] = new_bias
			#Calculate a new weight for each of the current neurons weights.
			for weight in range(network["neuron_counts"][layer - 1]):
				new_weight = _adjusted_weight(network, layer, neuron, weight)
				network["weights"][layer][neuron][weight] = new_weight

def _accumulate_errors(network: dict) -> None:
	'''
	Update the average error for each neuron with it current error (used
	in batch learning for each training example).
	'''
	#Update each neurons accumulated error with its current error.
	for layer in range(1, network["layer_count"]):
		for neuron in range(network["neuron_counts"][layer]):
			#Add current error to accumulated error.
			error = network["errors"][layer][neuron]
			network["accumulated_errors"][layer][neuron] += error
			#Divide by two to keep a running average.
			#Divide if this is not the first error accumulated.
			if network["accumulated_errors"][layer][neuron] != error:
				network["accumulated_errors"][layer][neuron] /= 2.0

def _use_accumulated_errors(network: dict) -> None:
	'''
	Set the error for each neuron to its average error (used in batch
	learning after each epoch before adjusting the network).
	'''
	#Set each nurons current error to its accumulated error.
	for layer in range(1, network["layer_count"]):
		for neuron in range(network["neuron_counts"][layer]):
			error = network["accumulated_errors"][layer][neuron]
			network["errors"][layer][neuron] = error

def _reset_accumulated_errors(network: dict) -> None:
	'''
	Reset the average error for each neuron back to zero (used in batch
	learning after adjusting the network).
	'''
	#Reset each nurons accumulated error back to zero.
	for layer in range(1, network["layer_count"]):
		for neuron in range(network["neuron_counts"][layer]):
			network["accumulated_errors"][layer][neuron] = 0.0

####################
# Network Training #
####################

def _learn_online(network: dict, example_inputs: list[list[float]], expected_outputs: list[list[float]]) -> None:
	'''
	Run all training pairs through the network. Adjusting the network
	for each training pair.
	'''
	#Run each training pair though the network, adjusting each time.
	for inputs, expected in zip(example_inputs, expected_outputs):
		_forward_propagate(network, inputs)
		_backward_propagate(network, expected)
		_adjust_network(network)

def _learn_batch(network: dict, example_inputs: list[list[float]], expected_outputs: list[list[float]]) -> None:
	'''
	Run all training pairs through the network. Adjusting the network after
	all the pairs have been ran through.
	'''
	#Run each training pair though the network accumulating errors time.
	_reset_accumulated_errors(network)
	for inputs, expected in zip(example_inputs, expected_outputs):
		_forward_propagate(network, inputs)
		_backward_propagate(network, expected)
		_accumulate_errors(network)
	#Adjust based on accumulated errors.
	_use_accumulated_errors(network)
	_adjust_network(network)

def train_network(network: dict, example_inputs: list[list[float]], expected_outputs: list[list[float]]) -> None:
	'''
	Train the network with the given example inputs (each row is a set of
	network inputs) and the given expected outputs (each row is a set of
	network expected outputs). Each row in example inputs corresponds to
	the same row in expected outputs.
	'''
	#Use learning type for set number of epochs with given training data.
	print("\033[0K")
	for epoch in range(network["epochs"]):
		#Print current training percentage for user.
		percent = int((epoch / network["epochs"]) * 100)
		bar = "["
		for i in range(20):
			bar += "=" if i*5 <= percent else " "
		bar += "]"
		print("\033[1ATraining", bar, percent, "%")
		#Have network learn for this epoch.
		network["learning_type"](network, example_inputs, expected_outputs)
	print("\033[1A\033[2KTraining Complete")

###################
# Network Testing #
###################

def test_network_errors(network: dict, example_inputs: list[list[float]], expected_outputs: list[list[float]]) -> float:
	'''
	Simple way to measure the networks accuracy. Returns the accumulated
	output error across all the training pairs. Where 0.0 mean the network
	is 100% accurate. The lower the number the more accurate the network
	is.
	'''
	network_error = 0.0
	#Run each training pair though the network and get there errors.
	for inputs, expected in zip(example_inputs, expected_outputs):
		_forward_propagate(network, inputs)
		_backward_propagate(network, expected)
		#Calculate networks error so see how accurate it is now.
		for neuron in range(network["neuron_counts"][-1]):
			output = network["outputs"][-1][neuron]
			expected_output = expected[neuron]
			error = (expected_output - output)**2
			network_error += error
	return network_error

def test_network_outputs(network: dict, example_inputs: list[list[float]], expected_outputs: list[list[float]]) -> float:
	'''
	Simple way to measure the networks performance. Returns the how often
	the network is correct for the given training pairs (correct is when
	the nueron with the maximum activation value is the same in the output
	and the expected output). Where 1.0 means the network is 100% correct.
	The higher the number the more correct the network is.
	'''
	correct_outputs = 0
	#Run each training pair though the network.
	for inputs, expected in zip(example_inputs, expected_outputs):
		_forward_propagate(network, inputs)
		#Check if max in output matches max in expected.
		expected_max_index = expected.index(max(expected))
		outputs_max_index = network["outputs"][-1].index(max(network["outputs"][-1]))
		if expected_max_index == outputs_max_index:
			correct_outputs += 1
	#Calculate percentage correct.
	correct_outputs /= len(example_inputs)
	return correct_outputs

def use_network(network: dict, inputs: list[float]) -> list[float]:
	'''
	Returns the networks outputs for the given set of inputs.
	'''
	#Use inputs on network.
	_forward_propagate(network, inputs)
	#Output layer neuron outputs are the outputs. Copy and return them.
	outputs = [0] * network["neuron_counts"][-1]
	for neuron in range(network["neuron_counts"][-1]):
		output = network["outputs"][-1][neuron]
		outputs[neuron] = output
	return outputs

#################################################
# Exmaple Code For How To Use External Funtions #
#################################################

#This example trains the network to add two unique one digit numbers.
#
#	digit representation:   0  1  2  3  4  5  6  7  8  9
#	example input 2+3  ->  [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
#
#	digit representation:   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17
#	expected output 5  ->  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

#Create training data set.
example_inputs = []
expected_outputs = []
for left_digit in range(10):
	for right_digit in range(10):
		if left_digit == right_digit:
			continue
		example = [0] * 10
		example[left_digit] = 1
		example[right_digit] = 1
		example_inputs.append(example)
		expected = [0] * 18
		expected[left_digit + right_digit] = 1
		expected_outputs.append(expected)
#Create network.
seed(1) #Set seed for random number generation, for repeatable results.
input_neurons = 10
hidden_neurons = 14
hidden_layers = 2
output_neurons = 18
network = create_network(input_neurons, hidden_neurons, hidden_layers, output_neurons)
#Changed all configurable settings for this exmaple.
set_activation_function(network, "tanh")
set_learning_type(network, "online")
set_learning_rate(network, 0.18)
set_epochs(network, 4096)
#Train network.
train_network(network, example_inputs, expected_outputs)
#Test network.
print("Errors in training set:", test_network_errors(network, example_inputs, expected_outputs))
print("Correct answer percentage:", test_network_outputs(network, example_inputs, expected_outputs))
#print("outputs:", use_network(network, [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]))
