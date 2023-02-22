# Nerve_Net
Like its biological namesake this is a implementation of the simplest type of neural network, that being a feedforward network using backpropagation for learning.  This implementation uses only base python 3.10, no additional libraries required.

## Network Initialization
- Returns an untrained network with the given layer structure. Also sets default learning parameters.
  ```
  create_network(input_neurons: int,
                 hidden_neurons: int,
                 hidden_layers: int,
                 output_neurons: int) -> dict
  ```
- Sets the learning rate that will be used during train_network(). The learning rate controls how much the weights and bias will be chnaged to correct for neuron errors. Number should be less than one.
  ```
  set_learning_rate(network: dict,
                    learning_rate: float) -> None
  ```
- Sets the number of epochs that will be ran during train_network(). An epoch is training the network on the given data set once. 42 epochs would mean training the network on the given data set 42 times.
  ```
  set_epochs(network: dict,
             epochs: int) -> None
  ```
- Sets the activation function to be used by the networks neurons during train_network(). The activation function is used during forward propagation when calculating a given neurons output (activation). Activation functionssupported:
  - "sigmoid" - Classic activation functions.
  - "tanh" - Most the time better then sigmoid.
  ```
  set_activation_function(network: dict,
                          function_name: str) -> None
  ```
- Set the learning type that will be used during train_network(). The learning type dictates when the networks weights and bias are updated. Learning types supported:
    - "online" - weights and bias updated for every training example.
    - "batch" - weights and bias updated for every epoch.
  ```
  set_learning_type(network: dict,
                    learning_name: str) -> None
  ```

## Network Training
- Train the network with the given example inputs (each row is a set of network inputs) and the given expected outputs (each row is a set of network expected outputs). Each row in example inputs corresponds to the same row in expected outputs.
  ```
  train_network(network: dict,
                example_inputs: list[list[float]],
                expected_outputs: list[list[float]]) -> None
  ```

## Network Testing
- Simple way to measure the networks accuracy. Returns the accumulated output error across all the training pairs. Where 0.0 mean the network is 100% accurate. The lower the number the more accurate the network is.
  ```
  test_network_errors(network: dict,
                      example_inputs: list[list[float]],
                      expected_outputs: list[list[float]]) -> float
  ```
- Simple way to measure the networks performance. Returns the how often the network is correct for the given training pairs (correct is when the nueron with the maximum activation value is the same in the output and the expected output). Where 1.0 means the network is 100% correct. The higher the number the more correct the network is.
  ```
  test_network_outputs(network: dict,
                       example_inputs: list[list[float]],
                       expected_outputs: list[list[float]]) -> float
  ```
- Returns the networks outputs for the given set of inputs.
  ```
  use_network(network: dict,
              inputs: list[float]) -> list[float]
  ```

## Example Code
```
#This example trains the network to add two unique one digit numbers.
#
#       digit representation:   0  1  2  3  4  5  6  7  8  9
#       example input 2+3  ->  [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
#
#       digit representation:   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17
#       expected output 5  ->  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

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
set_learning_type(network, "batch")
set_learning_rate(network, 0.18)
set_epochs(network, 4096)

#Train network.
train_network(network, example_inputs, expected_outputs)

#Test network.
print("Errors in training set:", test_network_errors(network, example_inputs, expected_outputs))
print("Correct answer percentage:", test_network_outputs(network, example_inputs, expected_outputs))

#print("outputs:", use_network(network, [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]))
```
